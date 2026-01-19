"""Multi-agent orchestration patterns."""

from dataclasses import dataclass
from typing import Callable

from .base import Agent, Role


@dataclass
class AgentNode:
    """An agent with a name and optional routing logic."""

    name: str
    agent: Agent
    description: str = ""


class Orchestrator:
    """
    Coordinates multiple agents. Supports several patterns:

    1. Router: A coordinator agent decides which specialist to invoke
    2. Pipeline: Agents process sequentially, each building on the previous
    3. Parallel: Run multiple agents and aggregate results
    """

    def __init__(self, agents: list[AgentNode], coordinator: Agent | None = None):
        """
        Args:
            agents: List of specialist agents
            coordinator: Optional agent that decides routing (for router pattern)
        """
        self.agents = {a.name: a for a in agents}
        self.coordinator = coordinator

    def get_agent(self, name: str) -> Agent:
        """Get an agent by name."""
        return self.agents[name].agent

    def run_pipeline(self, initial_input: str, agent_order: list[str]) -> str:
        """
        Run agents in sequence, passing each output to the next.

        Args:
            initial_input: Starting prompt
            agent_order: List of agent names in execution order

        Returns:
            Final agent's response
        """
        current_input = initial_input

        for agent_name in agent_order:
            if agent_name not in self.agents:
                raise ValueError(f"Agent '{agent_name}' not found")

            agent = self.agents[agent_name].agent
            current_input = agent.run(current_input)
            agent.reset()  # Clear history for next use

        return current_input

    def run_parallel(
        self,
        prompt: str,
        agent_names: list[str] | None = None,
        aggregator: Callable[[dict[str, str]], str] | None = None,
    ) -> dict[str, str] | str:
        """
        Run multiple agents on the same prompt in parallel (conceptually).

        Args:
            prompt: Input to send to all agents
            agent_names: Which agents to run (default: all)
            aggregator: Optional function to combine results

        Returns:
            Dict of {agent_name: response} or aggregated string
        """
        names = agent_names or list(self.agents.keys())
        results = {}

        for name in names:
            if name not in self.agents:
                raise ValueError(f"Agent '{name}' not found")

            agent = self.agents[name].agent
            results[name] = agent.run(prompt)
            agent.reset()

        if aggregator:
            return aggregator(results)

        return results

    def run_routed(self, prompt: str) -> str:
        """
        Use coordinator agent to route to the appropriate specialist.

        The coordinator decides which agent(s) to invoke based on the prompt.
        Requires a coordinator agent to be set.
        """
        if not self.coordinator:
            raise ValueError("Router pattern requires a coordinator agent")

        # Build routing prompt with agent descriptions
        agent_list = "\n".join(
            f"- {name}: {node.description}" for name, node in self.agents.items()
        )

        routing_prompt = f"""Given the following request, determine which specialist agent should handle it.

Available agents:
{agent_list}

Request: {prompt}

Respond with ONLY the agent name that should handle this request."""

        # Get coordinator's routing decision
        chosen_name = self.coordinator.chat(routing_prompt).strip().lower()
        self.coordinator.reset()

        # Find matching agent (fuzzy match on name)
        for name in self.agents:
            if name.lower() in chosen_name or chosen_name in name.lower():
                return self.agents[name].agent.run(prompt)

        # Fallback: use first agent
        first_agent = list(self.agents.values())[0]
        return first_agent.agent.run(prompt)


def agent_as_tool(agent: Agent, name: str, description: str):
    """
    Wrap an agent as a tool that another agent can call.

    This enables hierarchical agent structures where a parent agent
    can delegate to child agents as needed.

    Usage:
        researcher = AnthropicAgent(system_prompt="You research topics...")
        writer = AnthropicAgent(system_prompt="You write articles...")

        # Main agent can call researcher as a tool
        main_agent = AnthropicAgent(
            system_prompt="You coordinate research and writing.",
            tools=[agent_as_tool(researcher, "research", "Research a topic")]
        )
    """
    from .tools import Tool

    def call_agent(query: str) -> str:
        response = agent.run(query)
        agent.reset()
        return response

    return Tool(
        name=name,
        description=description,
        fn=call_agent,
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The query to send to the agent"}
            },
            "required": ["query"],
        },
    )
