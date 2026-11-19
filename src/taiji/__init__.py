"""TAIJI: natural-language analytics over MCP services."""

from .client import Client
from .models import MCPConfig, ModelConfig, ModelError, QueryResult, ToolCall

__all__ = ["Client", "MCPConfig", "ModelConfig", "ModelError", "QueryResult", "ToolCall"]
__version__ = "0.1.0"
