from dataclasses import dataclass
from typing import Any, Callable, get_type_hints
import inspect
import json


@dataclass
class Tool:
    """Represents a callable tool for agents."""

    name: str
    description: str
    fn: Callable
    parameters: dict

    def to_openai_format(self) -> dict:
        """Format for OpenAI function calling."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def to_anthropic_format(self) -> dict:
        """Format for Anthropic tool use."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }


def tool(name: str | None = None, description: str | None = None):
    """Decorator to convert a function into a Tool."""

    def decorator(fn: Callable) -> Tool:
        tool_name = name or fn.__name__
        tool_description = description or fn.__doc__ or ""

        # Build JSON schema from type hints
        hints = get_type_hints(fn)
        sig = inspect.signature(fn)

        properties = {}
        required = []

        type_map = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object",
        }

        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue

            param_type = hints.get(param_name, str)
            json_type = type_map.get(param_type, "string")

            properties[param_name] = {"type": json_type}

            # Check for default values
            if param.default is inspect.Parameter.empty:
                required.append(param_name)

        parameters = {
            "type": "object",
            "properties": properties,
            "required": required,
        }

        return Tool(
            name=tool_name,
            description=tool_description.strip(),
            fn=fn,
            parameters=parameters,
        )

    return decorator


def tools_from_functions(functions: list[Callable]) -> list[Tool]:
    """Convert a list of functions to Tools using their docstrings."""
    return [tool()(fn) for fn in functions]
