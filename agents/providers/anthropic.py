import os
import json
from anthropic import Anthropic

from ..base import Agent, AgentResponse, Message, Role


class AnthropicAgent(Agent):
    """Agent using Anthropic's Claude API."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        system_prompt: str | None = None,
        tools: list | None = None,
        api_key: str | None = None,
        **kwargs,
    ):
        super().__init__(model, system_prompt, tools, **kwargs)
        self.client = Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))

    def _format_tools(self) -> list[dict]:
        return [t.to_anthropic_format() for t in self.tools]

    def _format_messages(self, messages: list[Message]) -> list[dict]:
        """Convert internal messages to Anthropic format."""
        formatted = []

        for msg in messages:
            if msg.role == Role.SYSTEM:
                continue  # System is passed separately

            if msg.role == Role.TOOL:
                formatted.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": msg.tool_call_id,
                        "content": msg.content,
                    }],
                })
            elif msg.tool_calls:
                # Assistant message with tool calls
                content = []
                if msg.content:
                    content.append({"type": "text", "text": msg.content})
                for tc in msg.tool_calls:
                    content.append({
                        "type": "tool_use",
                        "id": tc["id"],
                        "name": tc["name"],
                        "input": tc["arguments"],
                    })
                formatted.append({"role": "assistant", "content": content})
            else:
                formatted.append({
                    "role": msg.role.value,
                    "content": msg.content,
                })

        return formatted

    def _call_api(self, messages: list[Message]) -> AgentResponse:
        kwargs = {
            "model": self.model,
            "max_tokens": self.config.get("max_tokens", 4096),
            "messages": self._format_messages(messages),
        }

        if self.system_prompt:
            kwargs["system"] = self.system_prompt

        if self.tools:
            kwargs["tools"] = self._format_tools()

        response = self.client.messages.create(**kwargs)

        # Extract text content
        text_content = ""
        for block in response.content:
            if block.type == "text":
                text_content += block.text

        return AgentResponse(content=text_content, raw=response)

    def _parse_tool_calls(self, response: AgentResponse) -> list[dict]:
        tool_calls = []
        for block in response.raw.content:
            if block.type == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "name": block.name,
                    "arguments": block.input,
                })
        return tool_calls
