import os
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.anthropic import Anthropic
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.tools import ToolMetadata, QueryEngineTool

class BranchedRAG:
    def __init__(self, chroma_col_fin_stmt, chroma_col_stock_prices) -> None:
        self.chroma_col_fin_stmt = chroma_col_fin_stmt
        self.chroma_col_stock_prices = chroma_col_stock_prices
        self.embed = GeminiEmbedding(model_name="models/embedding-001", 
                                      api_key=os.environ.get("GOOGLE_API_KEY"))


    def generate_embeddings(self):
        index_fin_stmt = SimpleDirectoryReader.load_file("/data1/apple_FY24_Q3_Financial_Statements.pdf")
        index_stock_prices = SimpleDirectoryReader.load_file("/data1/SP 500 Stock Prices 2014-2017.csv")

        chroma_store_fin_stmt = ChromaVectorStore.from_collection(collection=index_fin_stmt)
        chroma_store_stock_prices = ChromaVectorStore.from_collection(collection=index_stock_prices)

        store_fin_stmt = VectorStoreIndex.from_vector_store(vector_store=chroma_store_fin_stmt, embed_model=self.embed)
        store_stock_prices = VectorStoreIndex.from_vector_store(vector_store=chroma_store_stock_prices, embed_model=self.embed)

        query_engine_fin_stmt = store_fin_stmt.as_query_engine()
        query_engine_stock_prices = store_stock_prices.as_query_engine()

        tools = [
            QueryEngineTool(query_engine=query_engine_fin_stmt, name="fin_stmt", metadata=ToolMetadata
                            (description="Use this tool for accessing financial statements")),
            QueryEngineTool(query_engine=query_engine_stock_prices, name="stock_prices", metadata=ToolMetadata
                            (description="Use this tool for accessing stock prices"))
        ]