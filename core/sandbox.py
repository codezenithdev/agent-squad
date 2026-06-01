"""Docker execution sandbox (v2.1).

Generated code is untrusted, so it runs **only** inside throwaway Docker
containers — never on the host. ``DockerSandbox.run`` mounts a workspace into a
container, runs one shell command with resource + time limits, captures the
real exit code / output, and the container is removed (``--rm``) afterward.

We shell out to the ``docker`` CLI rather than add a Docker SDK dependency. If
the daemon isn't reachable, ``docker_available()`` returns False and the agents
fall back to LLM-only assessment (so the pipeline still runs without Docker).

Network note: ``pip``/``npm`` installs need the network, so the default bridge
network is used. Isolation comes from the container boundary + memory/cpu/pids
limits + timeout + auto-removal (not from network lockdown).
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass
class SandboxResult:
    command: str
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out

    def tail(self, limit: int = 4000) -> str:
        """Combined output, trimmed to the last ``limit`` chars (errors live at
        the end of test output, so we keep the tail)."""
        combined = self.stdout
        if self.stderr.strip():
            combined += ("\n" if combined else "") + self.stderr
        combined = combined.strip()
        return combined if len(combined) <= limit else "..." + combined[-limit:]


@lru_cache(maxsize=1)
def docker_available() -> bool:
    """True if the Docker daemon is reachable. Cached (clear with
    ``docker_available.cache_clear()`` if the daemon starts mid-process)."""
    try:
        proc = subprocess.run(
            ["docker", "info"], capture_output=True, timeout=20, text=True
        )
        return proc.returncode == 0
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return False


class DockerSandbox:
    """Run shell commands against a workspace inside a one-shot container."""

    def __init__(
        self,
        image: str,
        *,
        memory: str = "1g",
        cpus: str = "2",
        pids_limit: int = 512,
        timeout: int = 300,
        network: str = "bridge",
    ) -> None:
        self.image = image
        self.memory = memory
        self.cpus = cpus
        self.pids_limit = pids_limit
        self.timeout = timeout
        self.network = network

    def run(self, workspace_dir: str, command: str, workdir: str = "/app") -> SandboxResult:
        """Mount ``workspace_dir`` at /app and run ``command`` via ``sh -lc``."""
        host = str(Path(workspace_dir).resolve())
        args = [
            "docker", "run", "--rm",
            "-v", f"{host}:/app",
            "-w", workdir,
            "--memory", self.memory,
            "--cpus", self.cpus,
            "--pids-limit", str(self.pids_limit),
            "--network", self.network,
            self.image,
            "sh", "-lc", command,
        ]
        try:
            proc = subprocess.run(
                args, capture_output=True, text=True, timeout=self.timeout
            )
            return SandboxResult(command, proc.returncode, proc.stdout, proc.stderr, False)
        except subprocess.TimeoutExpired as e:
            out = e.stdout or ""
            err = (e.stderr or "") + f"\n[sandbox: timed out after {self.timeout}s]"
            return SandboxResult(command, 124, _as_str(out), _as_str(err), True)
        except (FileNotFoundError, OSError, subprocess.SubprocessError) as e:
            return SandboxResult(command, 127, "", f"[sandbox error] {e}", False)


def _as_str(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value or ""
