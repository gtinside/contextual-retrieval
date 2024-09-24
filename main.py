import os
import anthropic
import chromadb
from dotenv import load_dotenv

class ContextualRetrieval:
    def __init__(self, anthropic_client, chroma_collection) -> None:
        self.anthropic_client = anthropic_client
        self.chroma_collection = chroma_collection


if __name__ == "__main__":
    load_dotenv()
    print(os.environ.get("ANTHROPIC_API_KEY"))
    chromadb_client = chromadb.Client()
    anthropic_client = anthropic.Anthropic()
    collection = chromadb_client.create_collection(name="embeddings")
    contextual_retrieval = ContextualRetrieval(anthropic_client=anthropic_client, chroma_collection=collection)
    



        




    