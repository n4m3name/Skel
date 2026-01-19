from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable
from enum import Enum


class Role(Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


@dataclass
class Message:
    role: Role
    content: str
    tool_call_id: str | None = None
    tool_calls: list[dict] | None = None

    def to_dict(self) -> dict:
        return {
            "role": self.role.value,
            "content": self.content,
            "tool_call_id": self.tool_call_id,
            "tool_calls": self.tool_calls,
        }


@dataclass
class ToolResult:
    tool_call_id: str
    result: Any
    error: str | None = None


@dataclass
class AgentResponse:
    content: str
    tool_calls: list[dict] | None = None
    raw: Any = None


class Agent(ABC):
    """Base agent interface for all providers."""

    def __init__(
        self,
        model: str,
        system_prompt: str | None = None,
        tools: list["Tool"] | None = None,
        **kwargs,
    ):
        self.model = model
        self.system_prompt = system_prompt
        self.tools = tools or []
        self.messages: list[Message] = []
        self.config = kwargs

    @abstractmethod
    def _call_api(self, messages: list[Message]) -> AgentResponse:
        """Provider-specific API call."""
        pass

    @abstractmethod
    def _format_tools(self) -> list[dict]:
        """Format tools for the specific provider."""
        pass

    @abstractmethod
    def _parse_tool_calls(self, response: AgentResponse) -> list[dict]:
        """Parse tool calls from provider response."""
        pass

    def add_message(self, role: Role, content: str, **kwargs) -> None:
        self.messages.append(Message(role=role, content=content, **kwargs))

    def run(self, user_input: str, max_iterations: int = 10) -> str:
        """Run the agent with automatic tool execution loop."""
        self.add_message(Role.USER, user_input)

        for _ in range(max_iterations):
            response = self._call_api(self.messages)

            tool_calls = self._parse_tool_calls(response)

            if not tool_calls:
                self.add_message(Role.ASSISTANT, response.content)
                return response.content

            # Execute tools and continue
            self.add_message(Role.ASSISTANT, response.content or "", tool_calls=tool_calls)

            for tool_call in tool_calls:
                result = self._execute_tool(tool_call)
                self.add_message(
                    Role.TOOL,
                    str(result.result) if not result.error else f"Error: {result.error}",
                    tool_call_id=result.tool_call_id,
                )

        return response.content

    def _execute_tool(self, tool_call: dict) -> ToolResult:
        """Execute a tool call and return the result."""
        tool_name = tool_call.get("name")
        tool_id = tool_call.get("id", "")
        arguments = tool_call.get("arguments", {})

        for tool in self.tools:
            if tool.name == tool_name:
                try:
                    result = tool.fn(**arguments)
                    return ToolResult(tool_call_id=tool_id, result=result)
                except Exception as e:
                    return ToolResult(tool_call_id=tool_id, result=None, error=str(e))

        return ToolResult(tool_call_id=tool_id, result=None, error=f"Tool '{tool_name}' not found")

    def chat(self, user_input: str) -> str:
        """Simple chat without tool execution (single turn)."""
        self.add_message(Role.USER, user_input)
        response = self._call_api(self.messages)
        self.add_message(Role.ASSISTANT, response.content)
        return response.content

    def reset(self) -> None:
        """Clear conversation history."""
        self.messages = []
