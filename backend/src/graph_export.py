"""
Graph export for LangGraph Studio.

LangGraph Studio expects a compiled graph as a module-level variable.
Run with: cd backend && langgraph dev
Opens: https://smith.langchain.com/studio/?baseUrl=http://localhost:2024
"""

from app.orchestration.deps import build_orchestration_deps
from app.orchestration.graph import build_graph

deps = build_orchestration_deps()
graph = build_graph(deps)
