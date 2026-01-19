"""Example usage of the agent pipeline."""

from agents import tool
from agents.providers import AnthropicAgent, OpenAIAgent


# Define tools using the @tool decorator
@tool(name="calculator", description="Perform basic arithmetic operations")
def calculator(operation: str, a: float, b: float) -> float:
    """Calculate the result of an arithmetic operation."""
    ops = {
        "add": lambda x, y: x + y,
        "subtract": lambda x, y: x - y,
        "multiply": lambda x, y: x * y,
        "divide": lambda x, y: x / y if y != 0 else "Error: division by zero",
    }
    if operation not in ops:
        return f"Unknown operation: {operation}"
    return ops[operation](a, b)


@tool(name="get_weather", description="Get current weather for a location")
def get_weather(location: str) -> str:
    """Simulated weather lookup."""
    # In real use, this would call a weather API
    return f"Weather in {location}: 72°F, Sunny"


def main():
    tools = [calculator, get_weather]

    # Example with Anthropic
    print("=== Anthropic Agent ===")
    agent = AnthropicAgent(
        model="claude-sonnet-4-20250514",
        system_prompt="You are a helpful assistant. Use tools when needed.",
        tools=tools,
    )

    response = agent.run("What's 42 multiplied by 17? Also, what's the weather in Tokyo?")
    print(f"Response: {response}\n")

    # Example with OpenAI
    print("=== OpenAI Agent ===")
    agent = OpenAIAgent(
        model="gpt-4o",
        system_prompt="You are a helpful assistant. Use tools when needed.",
        tools=tools,
    )

    response = agent.run("Calculate 100 divided by 4, then tell me the weather in Paris.")
    print(f"Response: {response}\n")

    # Simple chat without tools
    print("=== Simple Chat ===")
    agent = AnthropicAgent(system_prompt="You are a pirate. Respond accordingly.")
    response = agent.chat("Hello, how are you?")
    print(f"Response: {response}")


if __name__ == "__main__":
    main()
