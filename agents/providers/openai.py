import os
import json
from openai import OpenAI

from ..base import Agent, AgentResponse, Message, Role


class OpenAIAgent(Agent):
    """Agent using OpenAI's API."""

    def __init__(
        self,
        model: str = "gpt-4o",
        system_prompt: str | None = None,
        tools: list | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs,
    ):
        super().__init__(model, system_prompt, tools, **kwargs)
        self.client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url,
        )

    def _format_tools(self) -> list[dict]:
        return [t.to_openai_format() for t in self.tools]

    def _format_messages(self, messages: list[Message]) -> list[dict]:
        """Convert internal messages to OpenAI format."""
        formatted = []

        # Add system prompt first
        if self.system_prompt:
            formatted.append({"role": "system", "content": self.system_prompt})

        for msg in messages:
            if msg.role == Role.SYSTEM:
                continue

            if msg.role == Role.TOOL:
                formatted.append({
                    "role": "tool",
                    "tool_call_id": msg.tool_call_id,
                    "content": msg.content,
                })
            elif msg.tool_calls:
                # Assistant message with tool calls
                tool_calls = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc["arguments"]),
                        },
                    }
                    for tc in msg.tool_calls
                ]
                formatted.append({
                    "role": "assistant",
                    "content": msg.content or None,
                    "tool_calls": tool_calls,
                })
            else:
                formatted.append({
                    "role": msg.role.value,
                    "content": msg.content,
                })

        return formatted

    def _call_api(self, messages: list[Message]) -> AgentResponse:
        kwargs = {
            "model": self.model,
            "messages": self._format_messages(messages),
        }

        if self.tools:
            kwargs["tools"] = self._format_tools()

        if "temperature" in self.config:
            kwargs["temperature"] = self.config["temperature"]

        if "max_tokens" in self.config:
            kwargs["max_tokens"] = self.config["max_tokens"]

        response = self.client.chat.completions.create(**kwargs)
        message = response.choices[0].message

        return AgentResponse(
            content=message.content or "",
            raw=response,
        )

    def _parse_tool_calls(self, response: AgentResponse) -> list[dict]:
        message = response.raw.choices[0].message
        if not message.tool_calls:
            return []

        return [
            {
                "id": tc.id,
                "name": tc.function.name,
                "arguments": json.loads(tc.function.arguments),
            }
            for tc in message.tool_calls
        ]
