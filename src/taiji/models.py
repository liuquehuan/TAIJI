from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Any, Literal


class ModelError(RuntimeError):
    """The configured model could not return a valid response."""


@dataclass(frozen=True)
class ModelConfig:
    """OpenAI-compatible chat configuration. base_url may include /chat/completions."""

    model: str
    api_key: str = field(repr=False)
    base_url: str = "https://api.openai.com/v1"
    timeout: float = 120.0
    temperature: float = 0.0
    tool_mode: Literal["tools", "json"] = "tools"

    def __post_init__(self) -> None:
        if not self.model or not self.api_key or not self.base_url:
            raise ValueError("model, api_key and base_url must be nonempty")
        if self.timeout <= 0 or self.tool_mode not in ("tools", "json"):
            raise ValueError("Invalid timeout or tool_mode")

    @property
    def endpoint(self) -> str:
        base = self.base_url.rstrip("/")
        return base if base.endswith("/chat/completions") else base + "/chat/completions"

    @classmethod
    def from_env(cls, prefix: str = "OPENAI_") -> ModelConfig:
        """Read environment variables explicitly; importing taiji never loads .env files."""
        return cls(
            model=os.environ[prefix + "MODEL"],
            api_key=os.environ[prefix + "API_KEY"],
            base_url=os.getenv(prefix + "BASE_URL") or os.getenv(prefix + "API_BASE")
            or "https://api.openai.com/v1",
        )


@dataclass(frozen=True)
class MCPConfig:
    """Generic local stdio MCP server configuration."""

    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict, repr=False)
    cwd: str | None = None
    timeout: float = 600.0

    def __post_init__(self) -> None:
        if not self.command or self.timeout <= 0:
            raise ValueError("MCP command must be nonempty and timeout must be positive")


@dataclass
class ToolCall:
    server: str
    tool: str
    arguments: dict[str, Any]
    result: Any
    is_error: bool = False
    duration_seconds: float = 0.0


@dataclass
class QueryResult:
    answer: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    status: Literal["success", "partial", "failed"] = "success"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
