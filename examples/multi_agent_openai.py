"""Multi-agent orchestration examples using OpenAI."""

from agents import Orchestrator, AgentNode, agent_as_tool
from agents.providers import OpenAIAgent


def pipeline_example():
    """Sequential processing: researcher -> writer -> editor."""
    print("=== Pipeline Example ===\n")

    researcher = OpenAIAgent(
        system_prompt="You research topics and provide detailed factual information. Be thorough."
    )
    writer = OpenAIAgent(
        system_prompt="You take research notes and write clear, engaging content. Focus on readability."
    )
    editor = OpenAIAgent(
        system_prompt="You edit text for clarity, grammar, and conciseness. Return the polished version."
    )

    orchestrator = Orchestrator([
        AgentNode("researcher", researcher, "Researches topics"),
        AgentNode("writer", writer, "Writes content"),
        AgentNode("editor", editor, "Edits for clarity"),
    ])

    result = orchestrator.run_pipeline(
        "Write a brief explanation of how neural networks learn",
        agent_order=["researcher", "writer", "editor"]
    )

    print(f"Final output:\n{result}\n")


def parallel_example():
    """Multiple perspectives on the same question."""
    print("=== Parallel Example ===\n")

    technical = OpenAIAgent(
        system_prompt="You explain things from a technical/engineering perspective. Be precise."
    )
    creative = OpenAIAgent(
        system_prompt="You explain things using analogies and creative metaphors. Be imaginative."
    )

    orchestrator = Orchestrator([
        AgentNode("technical", technical),
        AgentNode("creative", creative),
    ])

    results = orchestrator.run_parallel("Explain how encryption works")

    print("Technical view:")
    print(results["technical"])
    print("\nCreative view:")
    print(results["creative"])
    print()


def hierarchical_example():
    """Main agent delegates to specialist sub-agents."""
    print("=== Hierarchical Example ===\n")

    # Specialist agents
    code_expert = OpenAIAgent(
        system_prompt="You are a coding expert. Provide code examples and technical explanations."
    )
    math_expert = OpenAIAgent(
        system_prompt="You are a math expert. Solve problems step by step."
    )

    # Main coordinator with sub-agents as tools
    main_agent = OpenAIAgent(
        system_prompt="""You are a helpful assistant. You have access to specialist agents:
- Use 'ask_coder' for programming questions
- Use 'ask_mathematician' for math problems
Delegate appropriately and synthesize their responses.""",
        tools=[
            agent_as_tool(code_expert, "ask_coder", "Ask the coding expert a programming question"),
            agent_as_tool(math_expert, "ask_mathematician", "Ask the math expert a math question"),
        ],
    )

    response = main_agent.run(
        "I need to calculate the factorial of 10, and also show me Python code to compute factorials."
    )

    print(f"Response:\n{response}\n")


if __name__ == "__main__":
    # Uncomment the example you want to run:
    # pipeline_example()
    # parallel_example()
    hierarchical_example()
