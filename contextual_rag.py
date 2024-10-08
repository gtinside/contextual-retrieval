import chromadb
from loguru import logger
from llama_index.llms.anthropic import Anthropic
from llama_index.core import Settings
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from dotenv import load_dotenv

"""
The contextual RAG can be implemented as follows:
1. Get chunks from the document
2. Use LLM to add context to each chunk
3. Generate the embeddings and store them in the vector database

"""

class ContextualRAG:

    def __init__(self, chroma_collection) -> None:
        self.chroma_collection = chroma_collection

    def generate_embeddings(self):
        # Step 1: Get the chunk
        logger.info("Getting the chunks")

        documents = SimpleDirectoryReader(input_files=["data/amazon-dynamo.pdf"]).load_data()
        nodes = SimpleNodeParser().get_nodes_from_documents(documents=documents)
        for node in nodes:
            logger.info(node)
        
        
        


        vector_store = ChromaVectorStore(chroma_collection = self.chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        VectorStoreIndex.from_documents(documents=documents, storage_context=storage_context)
        



if __name__ == "__main__":
    load_dotenv()
    Settings.llm = Anthropic(model="claude-3-sonnet-20240229")
    chroma_client = chromadb.EphemeralClient()
    collection = chroma_client.create_collection("data")
    contextual_rag = ContextualRAG(chroma_collection=collection)
    contextual_rag.generate_embeddings()
