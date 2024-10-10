import os
import chromadb
from dotenv import load_dotenv
from llama_index.core import Settings, StorageContext, VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.anthropic import Anthropic
from loguru import logger

class SimpleRAG:
    def __init__(self, chroma_collection, data_file) -> None:
        self.chroma_collection = chroma_collection
        self.embed_model = GeminiEmbedding(model_name="models/embedding-001", 
                                      api_key=os.environ.get("GOOGLE_API_KEY"))
        self.data_file = data_file

    
    def generate_embeddings(self):
        # Initialize Chromadb llamaindex wrapper class and Gemini embedding model
        chroma_vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
        
        # Get the documents ready to be loaded
        documents = SimpleDirectoryReader(input_files=[f"data/{self.data_file}"]).load_data()
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
    chromadb_client = chromadb.EphemeralClient()
    collection = chromadb_client.create_collection(name="embeddings", get_or_create=True)
    simple_rag = SimpleRAG(chroma_collection=collection, data_file="apple_FY24_Q3_Financial_Statements.pdf")
    simple_rag.generate_embeddings()
    query = "Earnings before taxes"
    logger.info("Query: {}, \n Response: {}", query, simple_rag.query_data(query))






                                                                                                                                                                                                                           