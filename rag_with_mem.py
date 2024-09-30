import os
import chromadb
from dotenv import load_dotenv
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.embeddings.gemini import GeminiEmbedding 
from llama_index.core import Settings, VectorStoreIndex, StorageContext, SimpleDirectoryReader
from llama_index.llms.anthropic import Anthropic
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.memory.chat_summary_memory_buffer import ChatSummaryMemoryBuffer

'''
The correct way to implement the RAG with context is as follows:
- For every user query, look in the chat history and determine if the current query is related.
- If it is then reword the current query to include summary from previous conversations
- The question and answer needs to be stored in cache for the next time

'''

class SimpleRAGWithMemory:
    def __init__(self, chroma_collection) -> None:
        self.chroma_collection = chroma_collection
        self.embed_model = GeminiEmbedding(model_name="models/embedding-001", 
                                      api_key=os.environ.get("GOOGLE_API_KEY"),
                                      title="Apple Q3 2024 Financial Statement")
        self.chat_memory = ChatSummaryMemoryBuffer.from_defaults(llm=Settings.llm)

    def generate_embeddings(self):
        documents = SimpleDirectoryReader("data").load_data()
        chroma_vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
        sc = StorageContext.from_defaults(vector_store=chroma_vector_store)
        VectorStoreIndex.from_documents(documents=documents, embed_model=self.embed_model, storage_context=sc)
        

    def query_data(self, query):
        chroma_vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
        index = VectorStoreIndex.from_vector_store(vector_store=chroma_vector_store, embed_model=self.embed_model)
        engine = index.as_query_engine()
        response= engine.query(query)
        
        # Storing data in memory
        self.chat_memory.put_messages([ChatMessage(role=MessageRole.USER, content=query)]
                                      , ChatMessage(role=MessageRole.ASSISTANT, content=response))
        return response
       


if __name__ == "__main__":
    load_dotenv()
    Settings.llm = Anthropic(model="claude-3-sonnet-20240229")
    chroma_client = chromadb.EphemeralClient()
    chroma_collection = chroma_client.create_collection("rag_with_mem")
    rag_with_mem = SimpleRAGWithMemory(chroma_collection)
    rag_with_mem.generate_embeddings()
    print(rag_with_mem.query_data("Total revenue"))

    
