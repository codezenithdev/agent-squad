"""The database agent — designs the data layer.

Reads the requirement plus the system/backend context and produces a concrete
``db_schema`` (tables, columns, keys, relationships, indexes). The backend
framework is passed in so the schema can reflect its conventions (e.g. ORM
naming), keeping the layers consistent.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage

from core.llm import complete
from core.state import AgentState
from core.tools import non_empty

DATABASE_SYSTEM = (
    "You are a senior database engineer. Produce a concrete relational schema: "
    "tables with columns and types, primary/foreign keys, relationships, and "
    "the indexes needed for the described access patterns. Be specific."
)


async def database(state: AgentState) -> dict:
    user = (
        f"Requirement:\n{state['input']}\n\n"
        f"Backend framework: {state.get('detected_backend_framework', 'unknown')}\n\n"
        f"System design:\n{state.get('system_design', '')}\n\n"
        "Produce the database schema."
    )
    schema = await complete("database", DATABASE_SYSTEM, user)
    non_empty(schema, "db_schema")

    return {
        "db_schema": schema,
        "messages": [AIMessage(content="[database] schema produced", name="database")],
    }
