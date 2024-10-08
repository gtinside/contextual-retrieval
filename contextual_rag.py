import chromadb
from loguru import logger

from llama_index.llms.anthropic import Anthropic
from llama_index.core import Settings, Document
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.llms import ChatMessage, MessageRole 
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
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
        documents = SimpleDirectoryReader(input_files=["data/amazon-dynamo.pdf"]).load_data()
        nodes = SimpleNodeParser().get_nodes_from_documents(documents=documents)
        logger.info("Total number of documents: {} and chunks: {}",len(documents), len(nodes))

        # Step 2: Read the PDF and prepare the prompt
        processed_documents = []
        for i, node in enumerate(nodes):
            logger.info("Processing {} of {} nodes", i, len(nodes))
            context_prompt = f"""
                <document> 
                {"\n".join([document.get_content() for document in documents])} 
                </document> 
                Here is the chunk we want to situate within the whole document 
                <chunk> 
                {node.get_content()} 
                </chunk> 
                Please give a short succinct context to situate this chunk within the overall document for the 
                purposes of improving search retrieval of the chunk. Answer only with the succinct context 
                and nothing else. """

            message = ChatMessage(role=MessageRole.USER, content=context_prompt)
            logger.info("Received response: {}", Settings.llm.chat(messages=[message]))
            processed_documents.append(Document(text=message))
        vector_store = ChromaVectorStore(chroma_collection = self.chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        VectorStoreIndex.from_documents(documents=processed_documents, storage_context=storage_context)
        



if __name__ == "__main__":
    load_dotenv()
    Settings.llm = Anthropic(model="claude-3-sonnet-20240229")
    Settings.embed_model = GeminiEmbedding(model_name="models/embedding-001")
    chroma_client = chromadb.EphemeralClient()
    collection = chroma_client.create_collection("data")
    contextual_rag = ContextualRAG(chroma_collection=collection)
    contextual_rag.generate_embeddings()
