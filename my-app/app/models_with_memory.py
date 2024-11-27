from json import tool
import os
from zoneinfo import ZoneInfo
from langchain_ollama import ChatOllama, OllamaEmbeddings
from llama_index.core import Settings
from llama_index.core import StorageContext
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.schema import ImageNode, TextNode
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.llms.langchain import LangChainLLM
from llama_index.core.tools import FunctionTool, ToolOutput
from llama_index.core.agent import ReActAgent, FunctionCallingAgentWorker
from llama_index.core.agent.react.formatter import ReActChatFormatter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
import requests
from transformers import CLIPProcessor, CLIPModel
import numpy as np
import chromadb
import base64
import fitz
from PIL import Image
import io
from typing import List, Dict, Optional, Tuple
import torch
from .config import settings
from .database import db_manager
from .timezone import timezone_service
from .chat_memory import chat_history


class EnhancedLLMService:
    def __init__(self):
        # Initialize Database MySQL
        try:
            db_manager.connect_database(database_name="db_AI")

        except Exception as e:
            print(f"Error connecting to database: {e}")

        # Initialize LLM models Llava for each image
        self.llm_llava = ChatOllama(model="llava:13b", base_url=settings.OLLAMA_URL)

        # Initialize LangChain Ollama
        self.chat_model = ChatOllama(
            model=settings.LLM_MODEL, base_url=settings.OLLAMA_URL
        )

        self.embedding_model = OllamaEmbeddings(
            model=settings.EMBED_MODEL, base_url=settings.OLLAMA_URL
        )

        # Initialize CLIP
        model_name = "openai/clip-vit-base-patch32"
        self.clip_model = CLIPModel.from_pretrained(model_name)
        self.clip_processor = CLIPProcessor.from_pretrained(model_name)

        # Get embedding dimension
        self.embedding_dim = 768  # Ollama dimension

        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=str(settings.CHROMA_DIR))

        # Create collections with specific dimensions
        try:
            # Try to delete existing collections
            self.chroma_client.delete_collection(f"{settings.COLLECTION_NAME}_text")
            self.chroma_client.delete_collection(f"{settings.COLLECTION_NAME}_images")
        except:
            pass

        # Create new collections with Text Collection
        self.text_collection = self.chroma_client.create_collection(
            name=f"{settings.COLLECTION_NAME}_text",
            metadata={"hnsw:space": "cosine", "dimension": self.embedding_dim},
        )

        # Create new collections with Image Collection
        self.image_collection = self.chroma_client.create_collection(
            name=f"{settings.COLLECTION_NAME}_images",
            metadata={"hnsw:space": "cosine", "dimension": self.embedding_dim},
        )

        # Create vector stores
        self.text_store = ChromaVectorStore(chroma_collection=self.text_collection)
        self.image_store = ChromaVectorStore(chroma_collection=self.image_collection)

        # Create index with storage context about Text and Image
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.text_store, image_store=self.image_store
        )

        # Initialize LlamaIndex components
        self.llm = LangChainLLM(llm=self.chat_model)
        self.embed_model = LangchainEmbedding(langchain_embeddings=self.embedding_model)

        # Configure settings
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model
        Settings.chunk_size = settings.CHUNK_SIZE
        Settings.chunk_overlap = settings.CHUNK_OVERLAP

        # Use initialized chat memory instance
        # self.chat_memory = chat_memory

        # Text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP
        )

        # Initialize index and retriever as None
        self.index = None
        self.retriever = None

        # Enhanced QA Prompt
        self.qa_prompt = PromptTemplate(
            template="""Based on the provided context, provide a comprehensive answer.

            Context:
            {context}

            Question: {question}

            Instructions:
            1. Reference specific page numbers when relevant
            2. Format your answer in a clear, structured way
            3. Be precise and concise in your response
            4. Focus on the most relevant information

            Answer:""",
            input_variables=["context", "question"],
        )

        # Initialize tools list
        self.tools_list = {}
        self.tools_list["get_faq_answer"] = self.get_faq_answer
        self.tools_list["get_weather"] = self.get_weather
        # Initialize timezone and faq
        self.setup_timezone_and_faq_tools()

        # Initialize agent
        self.setup_tools_and_agent()

    def setup_tools_and_agent(self):
        """Initialize tools and agent using LlamaIndex"""
        try:
            # Define tools using FunctionTool TimeZone
            timezone_tool = FunctionTool.from_defaults(
                fn=self.tools_list["get_time_timezone"],
                name="get_time_timezone",
                description="Get current time for a specific timezone (e.g., Asia/Tokyo). Only use when asked about time in specific timezone.",
            )

            # Define tools using FunctionTool Weather
            weather_tool = FunctionTool.from_defaults(
                fn=self.tools_list["get_weather"],
                name="get_weather",
                description="Get current weather information for a specific location. Only use when asked about weather conditions.",
            )

            # Define tools using FunctionTool Product Information
            product_tool = FunctionTool.from_defaults(
                fn=self.get_information_product,
                name="get_information_product",
                description="Get detailed product information by product name or code (e.g., K-1S, 化粧箱K-1, ｹｼｮｳﾊﾞｺｹｲｲﾁ). Only use when asked about specific products.",
            )

            # Define tools using FunctionTool PDF - Điều chỉnh mô tả chi tiết hơn
            pdf_tool = FunctionTool.from_defaults(
                fn=self._search_pdf_content,
                name="search_pdf",
                description="Search through PDF content for relevant information. MUST USE when: 1) Asked about any documentation or manuals, 2) Need to find specific text or content, 3) Questions about procedures/policies, 4) When product information alone is not sufficient.",
            )

            # Define conversation tool using chat_conversation function
            chat_tool = FunctionTool.from_defaults(
                fn=self.chat_conversation,
                name="general_chat",
                description="Handle general conversation, greetings, and casual queries. Use this for normal conversation, greetings, and when no other tools are needed.",
            )

            # Add chat_conversation to tools_list
            self.tools_list["general_chat"] = self.chat_conversation

            # Define system prompt for the agent
            system_prompt = """You are a helpful and friendly AI assistant named KSSupport. Your main purpose is to assist users with product information, document searches, and general inquiries.

            TOOLS AND THEIR USES (In Order of Priority):

            1. For product information (HIGHEST PRIORITY - use get_information_product first):
            - Always check product information first for any product-related queries
            - Use for any question about products, even partial matches
            - Must use for queries containing:
                * Product codes (e.g., K-1S, K-2S)
                * Product names (e.g., 化粧箱K-1, ｹｼｮｳﾊﾞｺｹｲｲﾁ)
                * Words like "product", "item", "what is", "tell me about"
            Example queries:
            - "What is K-1S?"
            - "Tell me about K-1S"
            - "K-1S specification"
            - "Show me product K-1S"
            - "What's the information for 化粧箱K-1?"
            - "ｹｼｮｳﾊﾞｺｹｲｲﾁ details"

            2. For document searches (EQUALLY HIGH PRIORITY - use search_pdf for any content queries):
            - MUST USE when users ask:
                * About any documentation, manuals, or guides
                * For specific text or content searches
                * About procedures, policies, or instructions
                * Questions starting with "where can I find", "how to", "what are the steps"
                * When looking for detailed explanations
                * When product information alone isn't enough
            Example queries:
            - "How do I process a return?"
            - "What are the steps for ordering?"
            - "Where can I find information about X?"
            - "What does the manual say about Y?"
            - "Tell me about the process of Z"
            - "Find information about product usage"
            - "Search for policy regarding X"
            - "List some rules of KSS company"

            3. For timezone queries (use get_time_timezone):
            - Only for specific timezone questions
            - Convert locations to timezone format (e.g., "Tokyo" → "Asia/Tokyo")

            4. For weather information (use get_weather):
            - Only for specific weather condition queries
            - When location is clearly specified

            5. For general conversation (LAST RESORT - only use general_chat if no other tools match):
            - Use only when no other tools are applicable
            - Handle basic greetings and general questions
            - Example queries:
                * "Hello!"
                * "How are you?"
                * "What can you do?"

            DECISION MAKING PRIORITY:
            1. ALWAYS check product information first:
            - For ANY mention of products or codes
            - Even for partial or unclear product references
            - When in doubt about a product query, use get_information_product

            2. If not product-related, check document content:
            - For any detailed information needs
            - For technical queries
            - For process or procedure questions

            3. For specific service queries:
            - Use timezone tool for time-related questions
            - Use weather tool for weather-related questions

            4. ONLY as last resort:
            - Use general_chat if no other tools are applicable
            - When query is clearly just casual conversation

            For unclear queries:
            1. First try get_information_product if might be product-related
            2. Then try search_pdf for possible document content
            3. Check if timezone or weather specific
            4. Only then default to general_chat

            Remember:
            - Products and document searches are TOP PRIORITY
            - Always check product information first for any possible product reference
            - Use search_pdf for detailed information needs
            - Only use general_chat when NO OTHER TOOLS apply
            - Keep responses professional and helpful"""

            # Initialize ReAct agent with updated tools and prompt
            self.agent = ReActAgent.from_tools(
                tools=[chat_tool, timezone_tool, weather_tool, product_tool, pdf_tool],
                llm=self.llm,
                verbose=True,
                system_prompt=system_prompt,
                formatter=ReActChatFormatter(),
            )

            print("LlamaIndex agent setup completed")

        except Exception as e:
            print(f"Error setting up LlamaIndex agent: {e}")
            raise

    def process_image(
        self, pil_image: Image.Image, analysis_text: str = None
    ) -> np.ndarray:
        """Process image và kết hợp cả CLIP embedding và analysis embedding"""
        try:
            # 1. Get CLIP embedding
            if pil_image.mode not in ["RGB", "L"]:
                pil_image = pil_image.convert("RGB")

            inputs = self.clip_processor(images=pil_image, return_tensors="pt")
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)

            clip_embedding = image_features.numpy()[0]
            clip_embedding = clip_embedding / np.linalg.norm(clip_embedding)

            # 2. Get analysis text embedding nếu có
            if analysis_text:
                analysis_embedding = np.array(
                    self.embedding_model.embed_query(analysis_text)
                )
                analysis_embedding = analysis_embedding / np.linalg.norm(
                    analysis_embedding
                )

                # 3. Combine embeddings (weighted average)
                combined_embedding = (0.5 * clip_embedding) + (0.5 * analysis_embedding)
                combined_embedding = combined_embedding / np.linalg.norm(
                    combined_embedding
                )

                return combined_embedding

            return clip_embedding

        except Exception as e:
            print(f"Error in process_image: {e}")
            return None

    def _analyze_image_with_llava(self, image_bytes: bytes) -> Optional[str]:
        """Analyze image using LLaVA with correct message format"""
        try:
            # Convert image to base64
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")

            # Create message with correct format
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """Analyze this technical image and tell me:
                            1. What type of content does it show?
                            2. Are there any command lines or code visible?
                            3. What are the key technical elements?
                            4. Describe the main information being shown? 
                            5. Describe what its action is.""",
                        },
                        {
                            "type": "image_url",
                            "image_url": f"data:image/jpeg;base64,{image_base64}",
                        },
                    ],
                }
            ]

            # Get response from LLaVA
            response = self.llm_llava.invoke(messages)
            return response.content

        except Exception as e:
            print(f"Error analyzing image with LLaVA: {str(e)}")
            return "Failed to analyze image"

    def extract_pdf_content(self, file_path: str) -> List:
        nodes = []
        try:
            doc = fitz.open(file_path)

            # Process text for all pages first
            text_nodes = self._extract_text_from_pdf(doc, file_path)
            nodes.extend(text_nodes)

            # PENDING TO LATER VERSION
            # Then process images for all pages
            # image_nodes = self._extract_images_from_pdf(doc, file_path)
            # nodes.extend(image_nodes)

            return nodes

        finally:
            if "doc" in locals():
                doc.close()

    def _extract_text_from_pdf(
        self, doc: fitz.Document, file_path: str
    ) -> List[TextNode]:
        """Extract text content from PDF and create text nodes"""
        text_nodes = []

        for page_num in range(len(doc)):
            page = doc[page_num]

            # Process text
            text = page.get_text("text")
            if text.strip():
                chunks = self.text_splitter.split_text(text)
                for chunk in chunks:
                    # Get text embedding from Ollama
                    text_embedding = self.embedding_model.embed_query(chunk)

                    # Create text node
                    text_node = TextNode(
                        text=chunk,
                        embedding=text_embedding,
                        metadata={
                            "page": page_num + 1,
                            "source": file_path,
                            "total_pages": len(doc),
                            "type": "text",
                        },
                    )
                    text_nodes.append(text_node)

        return text_nodes

    def _extract_images_from_pdf(
        self, doc: fitz.Document, file_path: str
    ) -> List[ImageNode]:
        """Extract images from PDF and create image nodes"""
        image_nodes = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            text_context = page.get_text("text")

            image_list = page.get_images(full=True)
            for img_idx, img in enumerate(image_list):
                try:
                    xref = img[0]
                    base_image = doc.extract_image(xref)

                    if base_image and base_image["image"]:
                        try:
                            # Process image như cũ
                            image_bytes = base_image["image"]
                            image_ext = base_image.get("ext", "png").lower()
                            pil_image = Image.open(io.BytesIO(image_bytes))

                            # Resize và convert như cũ
                            if pil_image.mode not in ["RGB", "L"]:
                                pil_image = pil_image.convert("RGB")

                            max_size = 800
                            if (
                                pil_image.size[0] > max_size
                                or pil_image.size[1] > max_size
                            ):
                                pil_image.thumbnail(
                                    (max_size, max_size), Image.Resampling.LANCZOS
                                )

                            # Save to base64
                            buffer = io.BytesIO()
                            pil_image.save(buffer, format=image_ext.upper())
                            img_str = base64.b64encode(buffer.getvalue()).decode()

                            # Get LLaVA analysis first
                            img_analysis = self._analyze_image_with_llava(image_bytes)

                            # Get combined embedding with analysis
                            image_embedding = self.process_image(
                                pil_image, img_analysis
                            )

                            if image_embedding is not None:
                                metadata = {
                                    "page": page_num + 1,
                                    "source": file_path,
                                    "analysis_context": img_analysis,
                                    "text_context": text_context,
                                    "format": image_ext,
                                    "type": "image",
                                    "width": pil_image.size[0],
                                    "height": pil_image.size[1],
                                }

                                # Store both embeddings in metadata
                                metadata["combined_embedding"] = True

                                image_node = ImageNode(
                                    image=img_str,
                                    embedding=image_embedding,  # Combined embedding
                                    metadata=metadata.copy(),
                                )
                                image_nodes.append(image_node)

                        except Exception as process_error:
                            print(f"Error processing image data: {process_error}")
                            continue

                except Exception as img_error:
                    print(f"Error extracting image: {img_error}")
                    continue

        return image_nodes

    async def process_pdf(
        self, file_path: str, filename: str = None
    ) -> tuple[int, int, int]:
        try:
            # Extract content
            nodes = self.extract_pdf_content(file_path)

            self.index = MultiModalVectorStoreIndex(
                nodes, storage_context=self.storage_context
            )

            # Create retriever
            self.retriever = self.index.as_retriever(
                similarity_top_k=10, image_similarity_top_k=3, similarity_cutoff=0.5
            )

            # Count nodes by type
            text_nodes = [n for n in nodes if isinstance(n, TextNode)]
            image_nodes = [n for n in nodes if isinstance(n, ImageNode)]

            return len(text_nodes), len(text_nodes), len(image_nodes)

        except Exception as e:
            print(f"Error in process_pdf: {e}")
            raise

    def query_pdf(self, query: str) -> str:
        """Get response from agent for given query"""
        try:
            # Save user query to history
            chat_history.add_message("user", query)

            # Check if there is a relevant FAQ answer first
            faq_answer = self.get_faq_answer(query)
            if faq_answer is not None:
                chat_history.add_message("assistant", faq_answer)
                return faq_answer

            # Get agent response
            agent_response = self.agent.stream_chat(query)

            # Extract answer from agent response
            answer = agent_response.response

            # Extract any tool outputs if available
            tool_outputs = []
            for source in agent_response.sources:
                if isinstance(source, ToolOutput):
                    tool_outputs.append(source.content)

            # Combine tool outputs if any
            if tool_outputs:
                answer = "\n".join(tool_outputs)

            # If answer is empty or whitespace only, use chat_conversation
            if not answer or answer.strip() == "":
                # Try conversation again
                chat_response = self.chat_conversation(query)

                # Verify if chat_response is not empty
                if chat_response and chat_response.strip():
                    chat_history.add_message("assistant", chat_response)
                    return chat_response
                else:
                    # If still empty, provide a default response
                    default_response = "I apologize, but I'm not sure how to answer that question. Could you please rephrase or ask something else?"
                    chat_history.add_message("assistant", default_response)
                    return default_response

            # Save existing answer to history
            chat_history.add_message("assistant", answer)
            return answer

        except Exception as e:
            print(f"Error in query_pdf: {e}")
            error_msg = f"Error processing query: {str(e)}"
            chat_history.add_message("system", error_msg)
            return error_msg

    # Using query with Text and Image Peding with version
    def query_pdf_with_image(self, query: str) -> tuple[str, List[Dict]]:
        try:
            # Get results using retriever
            retrieval_results = self.retriever.retrieve(query)

            # Separate text and image results
            text_results = []
            image_results = []

            for result in retrieval_results:
                if isinstance(result.node, ImageNode):
                    image_results.append(result.node)

                else:
                    text_results.append(result.node)

            # Create text context
            context_text = "\n\n".join(
                f"[Page {node.metadata['page']}] {node.text}" for node in text_results
            )

            # Process images với metadata từ ImageNode
            formatted_images = [
                {
                    "filename": f"{node.metadata['source']}_{node.metadata['page']}",  # Tạo filename từ source và page
                    "image": node.image,  # Base64 string đã được lưu trong ImageNode
                    "metadata": {
                        "format": node.metadata["format"],
                        "page": node.metadata["page"],
                        "width": node.metadata["width"],
                        "height": node.metadata["height"],
                        "relevance_score": 1.0,  # Thêm default score
                        "below_text": node.metadata.get(
                            "text_context", ""
                        ),  # Lấy text_context từ metadata
                        "analysis": node.metadata.get(
                            "text_context", ""
                        ),  # Sử dụng text_context làm analysis
                        "type": node.metadata["type"],  # Thêm type từ metadata gốc
                        "source": node.metadata[
                            "source"
                        ],  # Thêm source từ metadata gốc
                    },
                }
                for node in image_results
            ]

            # Generate answer
            answer = self.chat_model.invoke(
                self.qa_prompt.format(
                    context=context_text,
                    question=query,
                )
            ).content

            return answer, formatted_images

        except Exception as e:
            print(f"Error in query_pdf: {e}")
            raise

    def get_database_value(self, query: str, params=None):
        """Execute database query with parameters"""
        try:
            if params is not None:
                return db_manager.get_data(query, params)
            return db_manager.get_data(query)
        except Exception as e:
            print(f"Database error: {e}")
            return None

    def compute_similarity(
        self, embedding1: List[float], embedding2: List[float]
    ) -> float:
        """Compute cosine similarity between two embeddings"""
        try:
            # Convert to numpy arrays
            a = np.array(embedding1)
            b = np.array(embedding2)

            # Calculate cosine similarity
            cosine_similarity = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
            return float(cosine_similarity)

        except Exception as e:
            print(f"Error computing similarity: {e}")
            return 0.0

    def get_faq_answer(self, question: str) -> str:
        """FAQ answer search with PDF fallback - only return answer"""
        try:
            sql_query = """
                SELECT seq, parent_menu_number, child_menu_number,
                    question, answer, tag1, tag2, tag3
                FROM d_faq 
                ORDER BY parent_menu_number, child_menu_number, seq
            """
            # params = (question, question, question, question)
            results = self.get_database_value(sql_query)

            if results is not None and not results.empty:
                # Convert input question to embedding
                question_embedding = self.embedding_model.embed_query(question)
                best_match = None
                best_similarity = 0

                # Find best matching answer
                for _, row in results.iterrows():
                    faq_embedding = self.embedding_model.embed_query(row["question"])
                    similarity = self.compute_similarity(
                        question_embedding, faq_embedding
                    )

                    if similarity > best_similarity and similarity > 0.5:
                        best_similarity = similarity
                        best_match = row["answer"]

                if best_match:
                    return best_match

            # If no FAQ answer found, use agent's stream_chat
            return None

        except Exception as e:
            print(f"Error in get_faq_answer: {e}")
            return f"エラーが発生しました: {str(e)}"

    def get_information_product(self, question: str) -> str:
        """Information of Product return all of information value of product"""
        try:
            sql_query = """
                SELECT jan_code, goods_name, goods_name_canat, 
                    standard_name, department_code, quantity, 
                    capacity, trade_start_date, trade_stop_date,
                    discontinued_date, memo
                FROM m_goods 
                WHERE goods_name LIKE CONCAT('%',%s, '%')
                OR goods_name_canat LIKE CONCAT('%',%s, '%')
            """
            params = (question, question)
            results = self.get_database_value(sql_query, params)

            if results is not None and not results.empty:
                # Convert input question to embedding
                question_embedding = self.embedding_model.embed_query(question)
                best_match = None
                best_similarity = 0
                best_row = None

                # Find best matching answer
                for _, row in results.iterrows():
                    # Compare with good_name
                    good_name_embedding = self.embedding_model.embed_query(
                        row["goods_name"]
                    )
                    similarity = self.compute_similarity(
                        question_embedding, good_name_embedding
                    )

                    if similarity > best_similarity and similarity > 0.5:
                        best_similarity = similarity
                        best_match = row["goods_name"]
                        best_row = row

                    # Compare with good_name_canat
                    goods_name_canat_embedding = self.embedding_model.embed_query(
                        row["goods_name_canat"]
                    )
                    similarity = self.compute_similarity(
                        question_embedding, goods_name_canat_embedding
                    )

                    if similarity > best_similarity and similarity > 0.5:
                        best_similarity = similarity
                        best_match = row["goods_name"]
                        best_row = row

                    if best_match and best_row is not None:
                        # Format the response with detailed information
                        response = f"""Product Information:
    - JAN Code: {best_row['jan_code']}
    - Product Name: {best_row['goods_name']} ({best_row['goods_name_canat']})
    - Standard Name: {best_row['standard_name']}
    - Department Code: {best_row['department_code']}
    - Quantity: {best_row['quantity']}
    - Capacity: {best_row['capacity']}
    - Trade Period: {best_row['trade_start_date']} to {best_row['trade_stop_date'] or 'Present'}
    - Status: {'Discontinued' if best_row['discontinued_date'] else 'Active'}
    - Additional Info: {best_row['memo']}"""
                        return response

            # If no results found or no matching similarity, return message
            return f"I don't have any information about the product: {question}"

        except Exception as e:
            print(f"Error in get_information_product: {e}")
            return f"エラーが発生しました: {str(e)}"

    def setup_timezone_and_faq_tools(self):
        """Setup all tool list relate of those"""
        try:

            def get_time_for_zone(timezone_name: str) -> str:
                """Function để sử dụng làm tool

                Args:
                    timezone_name (str): Tên múi giờ

                Returns:
                    str: Thông tin thời gian dạng string
                """
                # Get infor of time in zone
                result = timezone_service.get_time_timezone(timezone_name)

                # format result
                return timezone_service.format_timezone_response(result)

            # Adding in tools_list
            self.tools_list["get_time_timezone"] = get_time_for_zone

        except Exception as e:
            print(f"Error setting up timezone tools: {e}")

    def _search_pdf_content(self, query: str) -> str:
        """Tool function để search và tóm tắt PDF content"""
        try:
            if self.index is None or self.retriever is None:
                return "No PDF has been processed yet. Please process a PDF first."

            # Get results using retriever
            results = self.retriever.retrieve(query)

            # Format text results
            text_context = []
            for result in results:
                if isinstance(result.node, TextNode):
                    text_context.append(f"{result.node.text}")

            # If no relevant content found, return a default message
            if not text_context:
                return "No relevant content found."

            # Combine all text into a single string
            context_text = "\n\n".join(text_context)

            # Get final answer
            answer = self.chat_model.invoke(
                self.qa_prompt.format(context=context_text, question=query)
            ).content

            return answer if answer else "No summary could be generated."

        except Exception as e:
            return f"Error searching and summarizing PDF: {str(e)}"

    def clear_chat_history(self):
        """Clear conversation history"""
        try:
            self.chat_memory.clear()
            print("Chat history cleared successfully")
        except Exception as e:
            print(f"Error clearing chat history: {e}")

    def get_weather(self, location: str) -> str:
        """
        Fetch current weather data using WeatherAPI.

        Args:
            location (str): City name, zip code, or coordinates (latitude,longitude).
            api_key (str): Your WeatherAPI API key.

        Returns:
            str: A formatted string containing the weather data.
        """
        # Base URL for WeatherAPI
        url = f"http://api.weatherapi.com/v1/current.json?key={settings.WEATHER_API_KEY}&q={location}"

        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise an HTTPError for bad responses
            data = response.json()

            # Extract relevant weather details
            location_name = data["location"]["name"]
            region = data["location"]["region"]
            country = data["location"]["country"]
            localtime = data["location"]["localtime"]
            temperature = data["current"]["temp_c"]
            condition = data["current"]["condition"]["text"]
            wind_speed = data["current"]["wind_kph"]
            humidity = data["current"]["humidity"]
            feels_like = data["current"]["feelslike_c"]

            return (
                f"Weather in {location_name}, {region}, {country}:\n"
                f"Time: {localtime}\n"
                f"Condition: {condition}\n"
                f"Temperature: {temperature}°C (Feels like {feels_like}°C)\n"
                f"Humidity: {humidity}%\n"
                f"Wind Speed: {wind_speed} km/h\n"
            )
        except Exception as err:
            return f"An error occurred: {err}"

    def chat_conversation(self, query: str) -> str:
        """Handle general conversation and casual queries"""
        try:
            # Prepare conversation prompt
            conversation_prompt = f"""As a helpful and friendly AI assistant, please respond to this query in a natural, conversational way. 
            If it's a greeting or casual conversation, respond appropriately. 
            If it's a question, try to provide helpful information.
            
            Previous conversation context (if any):
            {chat_history.get_recent(3)}
            
            Current query: {query}

            Guidelines:
            - Be friendly and professional
            - For greetings, respond warmly (e.g. "Hello!", "Good morning!")
            - For general questions, provide helpful information
            - If you don't know something, be honest about it
            - Keep responses natural and engaging
            - Use appropriate tone based on the query
            
            Response:"""

            # Get response from chat model
            response = self.chat_model.invoke(conversation_prompt).content
            return response

        except Exception as e:
            print(f"Error in chat_conversation: {e}")
            return "I apologize, but I'm having trouble processing that. Could you please try rephrasing?"


llm_manager = EnhancedLLMService()
