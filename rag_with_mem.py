import os
from loguru import logger
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

        # Before sending the response, check if previous conversations are related
        query_with_context = self.get_previous_summary(query)
        logger.info("Executing the query: {}", query_with_context)
        response= engine.query(query_with_context)
        
        # Storing data in memory
        self.chat_memory.put_messages([ChatMessage(role=MessageRole.USER, content=query)
                                      , ChatMessage(role=MessageRole.ASSISTANT, content=response)])
        return response

    def get_previous_summary(self, query):
        logger.info("Checking for previous conversations related to {}", query)
        #TODO: Ideally this should just be last few messages, pulling everything for now
        chat_history = self.chat_memory.get()
        if not chat_history: 
            return query
        previous_questions = ",".join([str(chat.content) for chat in chat_history if chat.role == MessageRole.USER])
        previous_answers = ",".join([str(chat.content) for chat in chat_history if chat.role == MessageRole.ASSISTANT])
        prompt = f'''Based on the question: {previous_questions} and answer: {previous_answers}, 
                 generate a standalone question for {query}'''
        logger.info("Prompt to be sent is {}", prompt)
        response = Settings.llm.chat([ChatMessage(role=MessageRole.USER, content=query)])
        logger.info("Received response from LLM on comparison: {}", response.message)
        return response.message
    

if __name__ == "__main__":
    load_dotenv()
    Settings.llm = Anthropic(model="claude-3-sonnet-20240229")
    chroma_client = chromadb.EphemeralClient()
    chroma_collection = chroma_client.create_collection("rag_with_mem")
    rag_with_mem = SimpleRAGWithMemory(chroma_collection)
    rag_with_mem.generate_embeddings()
    logger.info("Query 1: {}", rag_with_mem.query_data("Total marketable securities on June 29, 2024?"))
    logger.info("Query 2, {}", rag_with_mem.query_data("How about on September 2023?"))
    logger.info("Query 3, {}", rag_with_mem.query_data("What's the percentage change between them?"))

    
