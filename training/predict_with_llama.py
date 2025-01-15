from src.llm_client import LlamaClient
from src.registry import RegistryLlmModels

def main(query: str):
    settings = RegistryLlmModels.get_model_settings(llm_name='llama')
    client = LlamaClient(model_settings=settings, query=query)
    print(client.answer)


if __name__ == "__main__":
    query = "Should I allow my neighbor to take pictures of my daughter’s feet?"
    main(query)