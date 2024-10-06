import os
import chromadb
from dotenv import load_dotenv
from loguru import logger
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, Settings
from llama_index.llms.anthropic import Anthropic
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.tools import ToolMetadata, QueryEngineTool
from llama_index.core.agent import ReActAgent


class BranchedRAG:
    def __init__(self, chroma_col_fin_stmt, chroma_col_stock_prices) -> None:
        self.chroma_col_fin_stmt = chroma_col_fin_stmt
        self.chroma_col_stock_prices = chroma_col_stock_prices
        self.embed = GeminiEmbedding(model_name="models/embedding-001", 
                                      api_key=os.environ.get("GOOGLE_API_KEY"))


    def generate_embeddings(self):
        chroma_store_fin_stmt = ChromaVectorStore.from_collection(collection=self.chroma_col_fin_stmt)
        chroma_store_stock_prices = ChromaVectorStore.from_collection(collection=self.chroma_col_stock_prices)

        index_fin_stmt_document = SimpleDirectoryReader(input_files=["data/apple_FY24_Q3_Financial_Statements.pdf"]).load_data()
        index_stock_prices_document = SimpleDirectoryReader(input_files=["data/SP_500_Stock_Prices.csv"]).load_data()

        storage_context_fin_stmt = StorageContext.from_defaults(vector_store=chroma_store_fin_stmt)
        storage_context_stock_prices = StorageContext.from_defaults(vector_store=chroma_store_stock_prices)
        
        store_fin_stmt = VectorStoreIndex.from_documents(documents=index_fin_stmt_document, storage_context=storage_context_fin_stmt, embed_model=self.embed)
        store_stock_prices = VectorStoreIndex.from_documents(documents=index_stock_prices_document, storage_context=storage_context_stock_prices, embed_model=self.embed)

        query_engine_fin_stmt = store_fin_stmt.as_query_engine()
        query_engine_stock_prices = store_stock_prices.as_query_engine()

        tools = [
            QueryEngineTool(query_engine=query_engine_fin_stmt, metadata=ToolMetadata
                            (name="fin_stmt", description="Use this tool for accessing financial statements")),
            QueryEngineTool(query_engine=query_engine_stock_prices, metadata=ToolMetadata
                            (name="stock_prices", description="Use this tool for accessing stock prices"))
        ]

        agent = ReActAgent.from_tools(llm=Settings.llm, tools=tools)
        self.agent = agent
    
    def query_data(self, query):
        response = self.agent.chat(query)
        return str(response.response)

if __name__ == "__main__":
    load_dotenv()
    Settings.llm = Anthropic(model="claude-3-sonnet-20240229")
    db = chromadb.EphemeralClient()
    branched_rag = BranchedRAG(chroma_col_fin_stmt=db.create_collection(name="fin_stmt"), 
                               chroma_col_stock_prices=db.create_collection(name="stock_prices"))
    branched_rag.generate_embeddings()
    query = "Total earnings of Apple during Q3 2024 and the average stock price during that period"
    response = branched_rag.query_data(query=query)
    logger.info("Query: {}, \nResponse: {}", query, response)
    

        

