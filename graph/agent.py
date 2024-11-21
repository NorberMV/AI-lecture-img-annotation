from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from dotenv import load_dotenv

load_dotenv()

class Agent:
    def __init__(self, model_name="gpt-4o", temperature=0, tools=[], system_prompt=None):
        """
        Initializes an Agent with specified model, temperature, tools, and system prompt.

        :param model_name: The name of the GPT model to use (default: "gpt-4o")
        :param temperature: The randomness of the model's output (default: 0)
        :param tools: List of tools to bind to the model (default: [])
        :param system_prompt: The system message to initialize the conversation (default: None)
        """
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )
        self.llm_with_tools = self.llm.bind_tools(tools)
        self.system_message = SystemMessage(content=system_prompt) if system_prompt else None

# Example usage
# if __name__ == "__main__":
#     # Assuming object_detection_tool is defined elsewhere
#     agent = Agent(
#         model_name="gpt-4o",
#         temperature=0.7,
#         tools=[object_detection_tool],
#         system_prompt="This is the main assistant system prompt."
#     )
#
#     # Accessing attributes directly
#     print(agent.llm)  # This will show the ChatOpenAI instance
#     print(agent.llm_with_tools)  # This will show the model with tools
#     print(agent.system_message)  # This will show the system message if set