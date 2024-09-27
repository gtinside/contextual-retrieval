import os
import anthropic
import chromadb
from dotenv import load_dotenv
from llama_index.core import StorageContext, VectorStoreIndex, SimpleDirectoryReader, load_index_from_storage
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

class ContextualRetrieval:
    def __init__(self, anthropic_client, chroma_collection) -> None:
        self.anthropic_client = anthropic_client
        self.chroma_collection = chroma_collection

    
    def generate_embeddings(self):
        # Initialize Chromadb llamaindex wrapper class and Gemini embedding model
        chroma_vector_store = ChromaVectorStore(chroma_collection=collection)
        embed_model = GeminiEmbedding(model_name="models/embedding-001", 
                                      api_key=os.environ.get("GOOGLE_API_KEY"),
                                      title="Apple Q3 2024 Financial Statement")
        # Get the documents ready to be loaded
        documents = SimpleDirectoryReader("data").load_data()
        storage_context = StorageContext.from_defaults(vector_store=chroma_vector_store)
         # Load everhything
        index = VectorStoreIndex.from_documents(documents=documents, embed_model=embed_model, storage_context=storage_context)
        
    def query_data(self, query):
        storage_context = StorageContext.from_defaults(persist_dir="/tmp")
        loaded_index = load_index_from_storage(storage_context=storage_context)
        query_engine = loaded_index.as_query_engine()
        return query_engine.query("Earnings before taxes")


if __name__ == "__main__":                                                                  
    load_dotenv()
    # Initializations
    chromadb_client = chromadb.PersistentClient()
    anthropic_client = anthropic.Anthropic()
    collection = chromadb_client.create_collection(name="embeddings")
    contextual_retrieval = ContextualRetrieval(anthropic_client=anthropic_client, 
                                               chroma_collection=collection)
    contextual_retrieval.generate_embeddings()
    contextual_retrieval.query_data






                                                                                                                                                                                                                           