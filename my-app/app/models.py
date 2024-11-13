from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from .config import settings

class LLMService:
    def __init__(self):
        # Initialize LLM
        self.llm = ChatOllama(
            model=settings.LLM_MODEL,
            base_url=settings.OLLAMA_URL
        )
        
        # Initialize Embedding
        self.embedding = OllamaEmbeddings(
            model=settings.EMBED_MODEL,
            base_url=settings.OLLAMA_URL
        )
        
        # Initialize Text Splitter
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )
        
        # Initialize QA Prompt
        self.qa_prompt = PromptTemplate(
            template="""Use the context to answer the question. If unsure, say you don't know.
            Context: {context}
            Question: {question}
            Answer:""",
            input_variables=["context", "question"]
        )

    def get_answer(self, query: str) -> str:
        """Get direct answer from LLM"""
        response = self.llm.invoke(query)
        return response.content
    
    async def process_pdf(self, file_path: str, filename: str = None) -> tuple[int, int]:
        """Process PDF and store in vector DB"""
        try:
            # Load and split PDF
            docs = PDFPlumberLoader(file_path).load_and_split()
            chunks = self.splitter.split_documents(docs)
            
            # Store in vector DB
            vector_store = Chroma(
                embedding_function=self.embedding,
                collection_name="pdf_collection",
                client_settings={
                    "chroma_api_impl": "rest",
                    "chroma_server_host": settings.CHROMA_HOST,
                    "chroma_server_http_port": settings.CHROMA_PORT,
                    "chroma_server_ssl_enabled": True
                }
            )
            
            # Add documents to collection
            vector_store.add_documents(chunks)
            
            return len(docs), len(chunks)
            
        except Exception as e:
            print(f"Error in process_pdf: {str(e)}")
            raise
    
    def query_pdf(self, query: str) -> tuple[str, list]:
        """Query PDF content"""
        try:
            # Initialize vector store with remote connection
            vector_store = Chroma(
                embedding_function=self.embedding,
                collection_name="pdf_collection",
                client_settings={
                    "chroma_api_impl": "rest",
                    "chroma_server_host": settings.CHROMA_HOST,
                    "chroma_server_http_port": settings.CHROMA_PORT,
                    "chroma_server_ssl_enabled": True
                }
            )
            
            # Get relevant documents
            docs = vector_store.similarity_search(
                query,
                k=settings.MAX_RESULTS
            )
            
            if not docs:
                return "No relevant documents found.", []
            
            # Create context from documents
            context = "\n\n".join(doc.page_content for doc in docs)
            
            # Get answer from LLM
            response = self.llm.invoke(
                self.qa_prompt.format(context=context, question=query)
            )
            answer = response.content
            
            # Prepare sources
            sources = [{
                "source": doc.metadata.get("source", "Unknown"),
                "content": doc.page_content
            } for doc in docs]
            
            return answer, sources
            
        except Exception as e:
            print(f"Error in query_pdf: {str(e)}")
            raise

llm_manager = LLMService()