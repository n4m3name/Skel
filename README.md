# Agent Pipeline

A minimal, provider-agnostic Python framework for building AI agents with tool use and multi-agent orchestration.

## Installation

```bash
pip install -r requirements.txt
```

Set your API keys:
```bash
export ANTHROPIC_API_KEY="your-key"
export OPENAI_API_KEY="your-key"
```

## Quick Start

```python
from agents import tool
from agents.providers import AnthropicAgent

@tool(name="greet", description="Greet someone by name")
def greet(name: str) -> str:
    return f"Hello, {name}!"

agent = AnthropicAgent(
    system_prompt="You are a friendly assistant.",
    tools=[greet],
)

response = agent.run("Please greet Alice")
print(response)
```

## Core Concepts

### Agents

Agents wrap LLM providers with a consistent interface:

```python
from agents.providers import AnthropicAgent, OpenAIAgent

# Anthropic (Claude)
agent = AnthropicAgent(
    model="claude-sonnet-4-20250514",  # default
    system_prompt="You are helpful.",
    tools=[...],
    max_tokens=4096,
)

# OpenAI
agent = OpenAIAgent(
    model="gpt-4o",  # default
    system_prompt="You are helpful.",
    tools=[...],
    temperature=0.7,
)

# OpenAI-compatible APIs (Ollama, vLLM, etc.)
agent = OpenAIAgent(
    model="llama3",
    base_url="http://localhost:11434/v1",
)
```

**Methods:**
- `agent.run(prompt)` - Run with automatic tool execution loop
- `agent.chat(prompt)` - Single turn, no tool execution
- `agent.reset()` - Clear conversation history

### Tools

Define tools using the `@tool` decorator. Parameter schemas are auto-generated from type hints:

```python
from agents import tool

@tool(name="calculator", description="Perform math operations")
def calculator(operation: str, a: float, b: float) -> float:
    """
    Supported operations: add, subtract, multiply, divide
    """
    ops = {
        "add": lambda x, y: x + y,
        "subtract": lambda x, y: x - y,
        "multiply": lambda x, y: x * y,
        "divide": lambda x, y: x / y,
    }
    return ops[operation](a, b)

# Use with an agent
agent = AnthropicAgent(tools=[calculator])
```

**Supported types:** `str`, `int`, `float`, `bool`, `list`, `dict`

### Multi-Agent Orchestration

Three patterns for coordinating multiple agents:

#### 1. Pipeline (Sequential)

Agents process in order, each receiving the previous output:

```python
from agents import Orchestrator, AgentNode
from agents.providers import AnthropicAgent

researcher = AnthropicAgent(system_prompt="You research topics thoroughly.")
writer = AnthropicAgent(system_prompt="You write clear, concise summaries.")
editor = AnthropicAgent(system_prompt="You edit for grammar and clarity.")

orchestrator = Orchestrator([
    AgentNode("researcher", researcher, "Researches topics"),
    AgentNode("writer", writer, "Writes content"),
    AgentNode("editor", editor, "Edits content"),
])

result = orchestrator.run_pipeline(
    "Write about quantum computing",
    agent_order=["researcher", "writer", "editor"]
)
```

#### 2. Parallel

Run multiple agents on the same prompt:

```python
optimist = AnthropicAgent(system_prompt="You see the positive side of everything.")
pessimist = AnthropicAgent(system_prompt="You see potential problems in everything.")

orchestrator = Orchestrator([
    AgentNode("optimist", optimist),
    AgentNode("pessimist", pessimist),
])

# Get dict of responses
results = orchestrator.run_parallel("Should I start a business?")
# {"optimist": "...", "pessimist": "..."}

# Or aggregate results
def combine(results):
    return f"Pros: {results['optimist']}\n\nCons: {results['pessimist']}"

combined = orchestrator.run_parallel(
    "Should I start a business?",
    aggregator=combine
)
```

#### 3. Router

A coordinator agent decides which specialist handles each request:

```python
coordinator = AnthropicAgent(model="claude-haiku-4-20250514")

math_agent = AnthropicAgent(system_prompt="You solve math problems.", tools=[calculator])
writing_agent = AnthropicAgent(system_prompt="You help with writing tasks.")

orchestrator = Orchestrator(
    agents=[
        AgentNode("math", math_agent, "Handles math and calculations"),
        AgentNode("writing", writing_agent, "Handles writing and editing"),
    ],
    coordinator=coordinator,
)

# Coordinator routes to appropriate agent
result = orchestrator.run_routed("What's 42 * 17?")  # Routes to math
result = orchestrator.run_routed("Write a haiku")    # Routes to writing
```

#### 4. Agent as Tool (Hierarchical)

Wrap an agent as a tool for another agent:

```python
from agents import agent_as_tool

researcher = AnthropicAgent(system_prompt="You research topics in depth.")

main_agent = AnthropicAgent(
    system_prompt="You answer questions. Use the research tool for complex topics.",
    tools=[
        agent_as_tool(researcher, "research", "Research a topic in depth")
    ],
)

# main_agent can now delegate research tasks
response = main_agent.run("Tell me about black holes")
```

## Adding New Providers

Extend the `Agent` base class:

```python
from agents.base import Agent, AgentResponse, Message

class MyProviderAgent(Agent):
    def __init__(self, model, system_prompt=None, tools=None, **kwargs):
        super().__init__(model, system_prompt, tools, **kwargs)
        self.client = MyProviderClient()

    def _format_tools(self) -> list[dict]:
        # Convert self.tools to provider's format
        return [...]

    def _call_api(self, messages: list[Message]) -> AgentResponse:
        # Call the provider API
        response = self.client.chat(...)
        return AgentResponse(content=response.text, raw=response)

    def _parse_tool_calls(self, response: AgentResponse) -> list[dict]:
        # Extract tool calls from response
        # Return list of {"id": str, "name": str, "arguments": dict}
        return [...]
```

## Features

- **Provider-agnostic**: Consistent interface across Anthropic, OpenAI, and compatible APIs
- **Automatic tool execution**: `agent.run()` handles the tool call loop
- **Type-safe tools**: `@tool` decorator generates JSON schemas from type hints
- **Multi-agent patterns**: Pipeline, parallel, router, and hierarchical orchestration
- **Conversation memory**: Agents maintain message history (use `reset()` to clear)
- **Minimal dependencies**: Only `anthropic` and `openai` SDKs

## Limitations

- **No streaming**: Responses are returned complete, not streamed
- **No async**: All calls are synchronous (wrap in `asyncio.to_thread` if needed)
- **No persistence**: Conversation history is in-memory only
- **No retry logic**: API failures raise exceptions directly
- **Basic tool schemas**: Complex nested types require manual schema definition
- **No vision/multimodal**: Text-only (extend `_format_messages` for images)
- **Sequential parallel**: `run_parallel` runs agents sequentially despite the name (use threading/asyncio for true parallelism)

## Project Structure

```
agents/
├── __init__.py          # Public exports
├── base.py              # Agent ABC, Message, Role
├── tools.py             # @tool decorator, Tool class
├── orchestrator.py      # Multi-agent coordination
└── providers/
    ├── __init__.py
    ├── anthropic.py     # Claude integration
    └── openai.py        # OpenAI/compatible APIs
examples/
└── basic_agent.py       # Usage examples
```

## Examples

See `examples/basic_agent.py` for working code examples.

```bash
cd Skel
python examples/basic_agent.py
```
