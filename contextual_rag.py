import chromadb
from loguru import logger
from simple_rag import SimpleRAG
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
            response = Settings.llm.chat(messages=[message])
            logger.info("Received response: {}", response)
            processed_documents.append(Document(text=response.message.content))
        
        vector_store = ChromaVectorStore(chroma_collection = self.chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        VectorStoreIndex.from_documents(documents=processed_documents, storage_context=storage_context)
        
    
    def query_data(self, query):
        # load index, as it is stored in vector database as embeddings
        # Step 1: Load Chroma Vector Store
        chroma_vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
       # Step 2: Load Index
        index = VectorStoreIndex.from_vector_store(
            vector_store=chroma_vector_store
        )
        query_engine = index.as_query_engine()
        return query_engine.query(query)



if __name__ == "__main__":
    load_dotenv()
    # General Settings
    Settings.llm = Anthropic(model="claude-3-sonnet-20240229")
    Settings.embed_model = GeminiEmbedding(model_name="models/embedding-001")
    
    # Contextual Embeddings
    chroma_client_contextual = chromadb.PersistentClient()
    collection = chroma_client_contextual.create_collection("data", get_or_create=True)
    contextual_rag = ContextualRAG(chroma_collection=collection)
    if collection.count() == 0:
        logger.info("Generating contextual embeddings")
        contextual_rag.generate_embeddings()
    
    # Non-contextual embeddings
    chromadb_client_ep = chromadb.EphemeralClient()
    collection = chromadb_client_ep.create_collection(name="embeddings")
    simple_rag = SimpleRAG(chroma_collection=collection, data_file="amazon-dynamo.pdf")
    simple_rag.generate_embeddings()

    
    queries = ["What is this paper all about?", "What are some of the technical considerations made in the paper?"]
    for i, query in enumerate(queries):
        logger.info(f"\n\nQuery-{i+1} : {query}, \nContextual Response-{i+1}: {contextual_rag.query_data(query)}, \nNon-contextual Response-{i+1}: {simple_rag.query_data(query)}")
        
    
