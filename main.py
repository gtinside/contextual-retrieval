import os
import chromadb
from dotenv import load_dotenv
from llama_index.core import Settings, StorageContext, VectorStoreIndex, SimpleDirectoryReader, load_index_from_storage
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.anthropic import Anthropic

class ContextualRetrieval:
    def __init__(self, chroma_collection) -> None:
        self.chroma_collection = chroma_collection
        self.embed_model = GeminiEmbedding(model_name="models/embedding-001", 
                                      api_key=os.environ.get("GOOGLE_API_KEY"),
                                      title="Apple Q3 2024 Financial Statement")

    
    def generate_embeddings(self):
        # Initialize Chromadb llamaindex wrapper class and Gemini embedding model
        chroma_vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
        
        # Get the documents ready to be loaded
        documents = SimpleDirectoryReader("data").load_data()
        storage_context = StorageContext.from_defaults(vector_store=chroma_vector_store)
         # The following command with store the index into chroma for embedding
        VectorStoreIndex.from_documents(documents=documents, embed_model=self.embed_model, storage_context=storage_context)
       
        
    def query_data(self, query):
        # load index, as it is stored in vector database as embeddings
        # Step 1: Load Chroma Vector Store
        chroma_vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
       # Step 2: Load Index
        index = VectorStoreIndex.from_vector_store(
            vector_store=chroma_vector_store,
            embed_model=self.embed_model
        )
        query_engine = index.as_query_engine()
        return query_engine.query(query)


if __name__ == "__main__":                                                                  
    load_dotenv()
    # Initializations
    Settings.llm = Anthropic(model="claude-3-sonnet-20240229")
    chromadb_client = chromadb.PersistentClient()
    collection = chromadb_client.create_collection(name="embeddings", get_or_create=True)
    contextual_retrieval = ContextualRetrieval(chroma_collection=collection)
    if collection.count() ==0:
        print("generating embeddings")
        contextual_retrieval.generate_embeddings()
    print(contextual_retrieval.query_data("Earnings before taxes"))






                                                                                                                                                                                                                           