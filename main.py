import os
import anthropic
import chromadb
from dotenv import load_dotenv
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

class ContextualRetrieval:
    def __init__(self, anthropic_client, chroma_collection) -> None:
        self.anthropic_client = anthropic_client
        self.chroma_collection = chroma_collection

    
    def generate_embeddings(self):
        embed_model = GeminiEmbedding(model_name="models/embedding-001", 
                                      api_key=os.environ.get("GOOGLE_API_KEY"),
                                      title="Apple Q3 2024 Financial Statement")
        embeddings = embed_model.get_text_embedding


if __name__ == "__main__":
    load_dotenv()
    chromadb_client = chromadb.EphemeralClient()
    anthropic_client = anthropic.Anthropic()
    collection = chromadb_client.create_collection(name="embeddings")
    chroma_vector_store = ChromaVectorStore(chroma_collection=collection)
    index = Vec
    contextual_retrieval = ContextualRetrieval(anthropic_client=anthropic_client, chroma_collection=collection)




        




    