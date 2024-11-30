""" Description: This file contains the implementation of the `ollama` class.
This class is responsible for handling the interaction with the OpenAI API for 
language generation.
And it is compatible with all of the OpenAI Compatible endpoints, including Ollama, 
OpenAI, and more.
"""

from typing import Iterator
import json, random, string
from .llm_interface import LLMInterface
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage
from langchain_core.messages import SystemMessage, trim_messages

class LLM(LLMInterface):

    def __init__(
        self,
        base_url: str,
        model: str,
        system: str,
        callback=print,
        organization_id: str = "z",
        project_id: str = "z",
        llm_api_key: str = "z",
        verbose: bool = False,
    ):
        """
        Initializes an instance of the `ollama` class.

        Parameters:
        - base_url (str): The base URL for the OpenAI API.
        - model (str): The model to be used for language generation.
        - system (str): The system to be used for language generation.
        - callback [DEPRECATED] (function, optional): The callback function to be called after each API call. Defaults to `print`.
        - organization_id (str, optional): The organization ID for the OpenAI API. Defaults to an empty string.
        - project_id (str, optional): The project ID for the OpenAI API. Defaults to an empty string.
        - llm_api_key (str, optional): The API key for the OpenAI API. Defaults to an empty string.
        - verbose (bool, optional): Whether to enable verbose mode. Defaults to `False`.
        """

        self.base_url = base_url
        self.model = model
        self.system = system
        self.callback = callback
        self.memory = []
        self.verbose = verbose
        self.client = ChatOpenAI(
            openai_api_key=llm_api_key,
            openai_api_base=base_url,
            model=model,
            temperature=0.3,
        )
        self.trimmer = trim_messages(
                        max_tokens=8,
                        strategy="last",
                        token_counter=len,
                        include_system=True,
                        allow_partial=False,
                        start_on="human",
                    )

        self.__set_system(system)

        if self.verbose:
            self.__printDebugInfo()

    def __set_system(self, system):
        """
        Set the system prompt
        system: str
            the system prompt
        """
        self.system = system
        self.memory.append(SystemMessage(content=system))

    def __print_memory(self):
        """
        Print the memory
        """
        print("Memory:\n========\n")
        # for message in self.memory:
        memory_json = [{one.type:one.content} for one in self.memory]
        print(memory_json)
        print("\n========\n")

    def __printDebugInfo(self):
        print(" -- Base URL: " + self.base_url)
        print(" -- Model: " + self.model)
        print(" -- System: " + self.system)

    def chat_iter(self, prompt: str) -> Iterator[str]:
        self.memory.append(HumanMessage(content=prompt))

        if self.verbose:
            self.__print_memory()
            print(" -- Base URL: " + self.base_url)
            print(" -- Model: " + self.model)
            print(" -- System: " + self.system)
            print(" -- Prompt: " + prompt + "\n\n")

        chat_completion = []
        try:
            trimmed_messages = self.trimmer.invoke(self.memory)
            chat_completion = self.client.stream(trimmed_messages)
        except Exception as e:
            print("Error calling the chat endpoint: " + str(e))
            self.__printDebugInfo()
            return "Error calling the chat endpoint: " + str(e)

        # a generator to give back an iterator to the response that will store
        # the complete response in memory once the iteration is done
        def _generate_and_store_response():
            complete_response = ""
            for chunk in chat_completion:
                yield chunk.content
                complete_response += chunk.content

            self.memory.append(AIMessage(content=complete_response))

            def serialize_memory(memory, filename):
                memory_json = [{one.type:one.content} for one in memory]
                with open(filename, "w") as file:
                    json.dump(memory_json, file)

            serialize_memory(self.memory, "mem.json")
            return

        return _generate_and_store_response()

    def handle_interrupt(self, heard_response: str) -> None:
        if self.memory[-1].type == "ai":
            self.memory[-1].content = heard_response + "..."
        else:
            if heard_response:
                self.memory.append(AIMessage(content=heard_response + "..."))
        self.memory.append(SystemMessage(content="[Interrupted by user]"))


def test():
    llm = LLM(
        base_url="https://open.bigmodel.cn/api/paas/v4/", #http://10.0.8.238:2064//v1
        model="glm-4", #qwen2-instruct
        callback=print,
        system="you're a good assistant",
        organization_id="organization_id",
        project_id="project_id",
        llm_api_key="94e4f47dba49757cb61f4e61fa372eb5.qQNMgZ3dGULd12oj",
        verbose=True,
    )
    while True:
        print("\n>> (Press Ctrl+C to exit.)")
        chat_complet = llm.chat_iter(input(">> "))

        for chunk in chat_complet:
            if chunk:
                print(chunk, end="")


if __name__ == "__main__":
    test()
