"""
LLM Module
----------
Loads local LLM using Ollama.
"""

from langchain_ollama import ChatOllama


class LocalLLM:

    def __init__(
        self,
        model_name="qwen3:8b",
        temperature=0
    ):

        self.llm = ChatOllama(
            model=model_name,
            temperature=temperature
        )

    def generate_response(self, query, context):

        prompt = f"""
You are a helpful AI assistant.

Answer ONLY using the provided context.

If the answer is not present in the context,
say:
"I could not find the answer in the provided context."

Context:
{context}

Question:
{query}
"""

        response = self.llm.invoke(prompt)

        return response.content
