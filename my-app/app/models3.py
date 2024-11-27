from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from .config import settings
import chromadb
import fitz
import io
import base64
from PIL import Image
import os
from typing import List, Dict, Tuple
from datetime import datetime
from langchain_community.llms import LlamaCpp
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from typing import List, Dict, Optional, Union
import torch
from transformers import AutoProcessor, CLIPVisionModel


class LLMService:
    def __init__(self):
        # Initialize LLM
        self.llm = ChatOllama(model=settings.LLM_MODEL, base_url=settings.OLLAMA_URL)

        # Initialize Embedding
        self.embedding = OllamaEmbeddings(
            model=settings.EMBED_MODEL, base_url=settings.OLLAMA_URL
        )

        # Initialize Text Splitter
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP
        )

        # Initialize QA Prompt
        self.qa_prompt = PromptTemplate(
            template="""Based on the provided context and images, please provide a comprehensive answer to the question.

        Context:
        {context}

        Related Images:
        {image_contexts}

        Question: {question}

        Instructions:
        1. Analyze both text content and related images
        2. Reference specific pages and images when relevant
        3. If images are important, describe their content and relationship to the answer
        4. Provide clear connections between text and images
        5. Include page numbers and image locations in your explanation

        Answer:""",
            input_variables=["context", "image_contexts", "question"],
        )

        # Initialize ChromaDB local client
        self.chroma_client = chromadb.PersistentClient(path=str(settings.CHROMA_DIR))

        # Get or create collections for text and images
        self.text_collection = self.chroma_client.get_or_create_collection(
            name=settings.COLLECTION_NAME
        )
        self.image_collection = self.chroma_client.get_or_create_collection(
            name=f"{settings.COLLECTION_NAME}_images"
        )

        # Ensure image directory exists
        self.image_dir = os.path.join(settings.STORAGE_DIR, "images")
        os.makedirs(self.image_dir, exist_ok=True)

    # Extract PFD to Text and Images
    def extract_pdf_content(self, file_path: str) -> Dict:
        content = {"text": [], "images": []}

        try:
            doc = fitz.open(file_path)

            for page_num in range(len(doc)):
                page = doc[page_num]

                # Extract text (giữ nguyên phần này)
                text = page.get_text("text")
                if text.strip():
                    doc_obj = Document(
                        page_content=text,
                        metadata={
                            "page": page_num + 1,
                            "source": file_path,
                            "total_pages": len(doc),
                        },
                    )
                    content["text"].append(doc_obj)

                # Extract images với base64
                image_list = page.get_images(full=True)
                sd = page.get_textbox(page.get_image_bbox(xref))

                for img_idx, img in enumerate(image_list):
                    try:
                        xref = img[0]
                        base_image = doc.extract_image(xref)

                        if base_image:
                            image_bytes = base_image["image"]
                            image_ext = base_image["ext"]

                            # Convert to base64
                            image_base64 = base64.b64encode(image_bytes).decode("utf-8")

                            # Validate image
                            img_obj = Image.open(io.BytesIO(image_bytes))

                            content["images"].append(
                                {
                                    "image": image_base64,  # Lưu base64 thay vì bytes
                                    "image_bytes": image_bytes,  # Giữ lại bytes nếu cần
                                    "page": page_num + 1,
                                    "index": img_idx,
                                    "width": img_obj.width,
                                    "height": img_obj.height,
                                    "format": image_ext,
                                    "xref": xref,
                                }
                            )

                    except Exception as img_error:
                        print(
                            f"Error extracting image {img_idx} from page {page_num + 1}: {str(img_error)}"
                        )
                        continue

            return content

        except Exception as e:
            print(f"Error extracting PDF content: {str(e)}")
            raise
        finally:
            if "doc" in locals():
                doc.close()

    # Save image to folder source local
    def save_image(self, img_data: Dict, page: int, source: str) -> Dict:
        """Save image and return metadata with base64"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"page_{page}_{timestamp}.{img_data['format']}"
            file_path = os.path.join(self.image_dir, filename)

            Image.open(io.BytesIO(img_data["image_bytes"])).save(file_path)

            return {
                "filename": filename,
                "path": file_path,
                "page": page,
                "source": source,
                "size": {"width": img_data["width"], "height": img_data["height"]},
                "format": img_data["format"],
                "file_size": os.path.getsize(file_path),
                "timestamp": timestamp,
                "image_base64": img_data.get("image", ""),
            }
        except Exception as e:
            print(f"Error saving image: {str(e)}")
            raise

    # Get a Image
    def get_image(self, filename: str) -> bytes:
        """Get image data from disk"""
        try:
            file_path = os.path.join(self.image_dir, filename)
            if os.path.exists(file_path):
                with open(file_path, "rb") as f:
                    return f.read()
            return None
        except Exception as e:
            print(f"Error getting image: {str(e)}")
            raise

    # Get answer of AI
    def get_answer(self, query: str) -> str:
        """Get direct answer from LLM"""
        response = self.llm.invoke(query)
        return response.content

    # Split data of file pdf get text and image
    async def process_pdf(
        self, file_path: str, filename: str = None
    ) -> tuple[int, int, int]:
        """Process PDF content with improved image handling"""
        try:
            # Extract and process content
            pdf_content = self.extract_pdf_content(file_path)
            text_chunks = self.splitter.split_documents(pdf_content["text"])

            # Store text chunks
            vector_store = Chroma(
                persist_directory=str(settings.CHROMA_DIR),
                embedding_function=self.embedding,
                collection_name=settings.COLLECTION_NAME,
            )
            vector_store.add_documents(text_chunks)

            # Map text chunks to pages
            page_text_map = {}
            for chunk in text_chunks:
                page = chunk.metadata["page"]
                page_text_map.setdefault(page, []).append(chunk.page_content)

            # Process images
            saved_images = []
            for img_data in pdf_content["images"]:
                try:
                    page = img_data["page"]
                    page_text = "\n".join(page_text_map.get(page, []))

                    image_info = self.save_image(img_data, page, filename or file_path)

                    # Store in ChromaDB with enhanced metadata
                    self.image_collection.add(
                        documents=[image_info["filename"]],
                        metadatas=[
                            {
                                "page": page,
                                "source": filename or file_path,
                                "width": image_info["size"]["width"],
                                "height": image_info["size"]["height"],
                                "format": image_info["format"],
                                "file_size": image_info["file_size"],
                                "timestamp": image_info["timestamp"],
                                "xref": img_data["xref"],
                                "type": "image",
                                "image_base64": image_info["image_base64"],
                                "page_text": page_text,
                                "text_context": self._get_surrounding_text(
                                    page_text, img_data["index"], 200
                                ),
                            }
                        ],
                        ids=[f"img_{page}_{img_data['index']}"],
                    )
                    saved_images.append(image_info)
                except Exception as e:
                    print(f"Error processing image: {str(e)}")

            vector_store.persist()
            return len(pdf_content["text"]), len(text_chunks), len(saved_images)

        except Exception as e:
            print(f"Error in process_pdf: {str(e)}")
            raise

    def _get_surrounding_text(self, text: str, position: int, window: int = 200) -> str:
        """Get text surrounding a position with given window size"""
        start = max(0, position - window)
        end = min(len(text), position + window)
        return text[start:end]

    # Get awser with pdf
    def query_pdf(self, query: str) -> tuple[str, Dict]:
        try:
            # Get text matches
            vector_store = Chroma(
                persist_directory=str(settings.CHROMA_DIR),
                embedding_function=self.embedding,
                collection_name=settings.COLLECTION_NAME,
            )

            # Get relevant documents
            docs = vector_store.similarity_search(query, k=settings.MAX_RESULTS)
            if not docs:
                return "No relevant information found.", None

            # Extract relevant pages
            relevant_pages = set()
            for doc in docs:
                page = doc.metadata.get("page")
                if page:
                    relevant_pages.add(page)

            # Get all images and find most relevant
            all_images = self.image_collection.get()
            most_relevant_image = None
            highest_score = -1

            if all_images and len(all_images["documents"]) > 0:
                for i, filename in enumerate(all_images["documents"]):
                    metadata = all_images["metadatas"][i]
                    page = metadata.get("page")
                    text_context = metadata.get("text_context", "").lower()
                    page_text = metadata.get("page_text", "").lower()

                    # Calculate relevance score
                    relevance_score = 0.0
                    query_terms = query.lower().split()

                    if page in relevant_pages:
                        relevance_score += 0.3
                    if any(term in text_context for term in query_terms):
                        relevance_score += 0.4
                    if any(term in page_text for term in query_terms):
                        relevance_score += 0.3

                    # Update most relevant image if score is higher
                    if relevance_score > highest_score:
                        highest_score = relevance_score
                        most_relevant_image = {
                            "filename": filename,
                            "image": metadata.get("image_base64", ""),
                            "metadata": {
                                "file_path": metadata.get("file_path"),
                                "file_size": metadata.get("file_size"),
                                "created_at": metadata.get("created_at"),
                                "modified_at": metadata.get("modified_at"),
                                "page": metadata.get("page"),
                                "timestamp": metadata.get("timestamp"),
                                "width": metadata.get("width"),
                                "height": metadata.get("height"),
                                "format": metadata.get("format"),
                            },
                        }

            # Prepare context for LLM
            context_text = "\n\n".join(
                f"[Page {doc.metadata.get('page')}] {doc.page_content}" for doc in docs
            )

            # Get answer from LLM
            answer = self.llm.invoke(
                self.qa_prompt.format(
                    context=context_text,
                    image_contexts=(
                        "Image available."
                        if most_relevant_image
                        else "No relevant image found."
                    ),
                    question=query,
                )
            ).content

            return answer, most_relevant_image

        except Exception as e:
            print(f"Error in query_pdf: {str(e)}")
            raise

    # Get All Image in Store Images
    def get_storage_images(self) -> Dict:
        """
        Get all images from storage/images directory

        Returns:
            Dict: {
                "total": int,
                "images": List[Dict] containing image info and base64 data
            }
        """
        try:
            # Get list of all files in the images directory
            image_files = [
                f
                for f in os.listdir(self.image_dir)
                if os.path.isfile(os.path.join(self.image_dir, f))
                and f.lower().endswith((".png", ".jpg", ".jpeg", ".gif"))
            ]

            images = []
            for filename in image_files:
                try:
                    # Get full file path
                    file_path = os.path.join(self.image_dir, filename)

                    # Read image file
                    with open(file_path, "rb") as img_file:
                        image_data = img_file.read()

                    # Convert to base64
                    image_base64 = base64.b64encode(image_data).decode("utf-8")

                    # Get file info
                    file_stats = os.stat(file_path)

                    # Try to get image dimensions
                    try:
                        with Image.open(file_path) as img:
                            width, height = img.size
                    except:
                        width, height = None, None

                    # Try to get metadata from filename
                    # Assuming filename format: page_1_20241114_113045.png
                    parts = filename.split("_")
                    page_number = int(parts[1]) if len(parts) > 1 else None
                    timestamp = (
                        "_".join(parts[2:]).split(".")[0] if len(parts) > 2 else None
                    )

                    # Create image info
                    image_info = {
                        "filename": filename,
                        "image": image_base64,
                        "metadata": {
                            "file_path": file_path,
                            "file_size": file_stats.st_size,
                            "created_at": datetime.fromtimestamp(
                                file_stats.st_ctime
                            ).isoformat(),
                            "modified_at": datetime.fromtimestamp(
                                file_stats.st_mtime
                            ).isoformat(),
                            "page": page_number,
                            "timestamp": timestamp,
                            "width": width,
                            "height": height,
                            "format": filename.split(".")[-1].lower(),
                        },
                    }

                    images.append(image_info)

                except Exception as e:
                    print(f"Error processing image {filename}: {str(e)}")
                    continue

            # Sort images by page number if available
            images.sort(
                key=lambda x: (x["metadata"]["page"] or float("inf"), x["filename"])
            )

            return {"total": len(images), "images": images}

        except Exception as e:
            print(f"Error getting storage images: {str(e)}")
            return {"total": 0, "images": []}


llm_manager = LLMService()
