import chromadb
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from dotenv import load_dotenv

class ContextualRAG:

    def __init__(self, chroma_collection) -> None:
        self.chroma_collection = chroma_collection

    def generate_embeddings(self):
        document = SimpleDirectoryReader(input_files=["data/amazon-dynamo.pdf"]).load_data()
        vector_store = ChromaVectorStore(chroma_collection = self.chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        VectorStoreIndex.from_documents(documents=document, storage_context=storage_context)
        



if __name__ == "__main__":
    load_dotenv()
    chroma_client = chromadb.EphemeralClient()
    chroma_client.create_collection("data")
    contextual_rag = ContextualRAG()
