from langchain_ollama import ChatOllama, OllamaEmbeddings
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
from typing import List, Dict, Tuple, Optional
from datetime import datetime


class LLMService:

    def __init__(self, storage_dir: str = "storage"):
        # Initialize LLM models
        self.llm = ChatOllama(model="llava:13b", base_url=settings.OLLAMA_URL)
        # Use LLaVA for image analysis
        self.text_llm = ChatOllama(
            model=settings.LLM_MODEL, base_url=settings.OLLAMA_URL
        )

        # Initialize Embedding
        self.embedding = OllamaEmbeddings(
            model=settings.EMBED_MODEL, base_url=settings.OLLAMA_URL
        )

        # Initialize Text Splitter
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP
        )

        # Initialize prompts
        self._setup_prompts()

        # Setup storage
        self.storage_dir = storage_dir
        self.image_dir = os.path.join(storage_dir, "images")
        os.makedirs(self.image_dir, exist_ok=True)

        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=storage_dir)
        self.text_collection = self.chroma_client.get_or_create_collection(
            name=settings.COLLECTION_NAME
        )
        self.image_collection = self.chroma_client.get_or_create_collection(
            name=f"{settings.COLLECTION_NAME}_images"
        )

    def _setup_prompts(self):
        """Setup various prompts for different tasks"""
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

        self.image_analysis_prompt = """Analyze this image in detail:
            1. Describe what you see
            2. Identify key technical elements
            3. Explain any text or numbers visible
            4. Note any important patterns or relationships
            5. Connect it to technical documentation context"""

    def extract_pdf_content(self, file_path: str) -> Dict:
        """Extract both text and images from PDF with improved context awareness"""
        content = {"text": [], "images": []}
        try:
            doc = fitz.open(file_path)
            img_count = 0  # Counter for debugging

            for page_num in range(len(doc)):
                page = doc[page_num]
                page_rect = page.rect

                print(f"Processing page {page_num + 1}")  # Debug log

                # Extract text with structure preservation
                try:
                    text = page.get_text("text")
                    if text.strip():
                        doc_obj = Document(
                            page_content=text,
                            metadata={
                                "page": page_num + 1,
                                "source": file_path,
                                "total_pages": len(doc),
                                "type": "text",
                            },
                        )
                        content["text"].append(doc_obj)
                        print(f"Extracted text from page {page_num + 1}")
                except Exception as text_error:
                    print(
                        f"Error extracting text from page {page_num + 1}: {str(text_error)}"
                    )

                # Get all images on the page
                image_list = page.get_images()
                print(
                    f"Found {len(image_list)} images in page {page_num + 1}"
                )  # Debug log

                # Process each image
                for img_idx, img in enumerate(image_list):
                    try:
                        xref = img[0]  # xref of image
                        print(
                            f"Processing image {img_idx + 1} with xref {xref}"
                        )  # Debug log

                        # Extract base image
                        base_image = doc.extract_image(xref)
                        if not base_image:
                            print(f"Could not extract image with xref {xref}")
                            continue

                        # Get image properties
                        image_bytes = base_image["image"]
                        image_rect = None

                        # Try multiple methods to get image location
                        # Method 1: Direct from page
                        try:
                            image_rect = page.get_image_bbox(xref)
                            print(
                                f"Found image rect using get_image_bbox: {image_rect}"
                            )
                        except Exception as e:
                            print(f"Error getting image bbox: {str(e)}")

                        # Method 2: Search in page dictionary
                        if not image_rect:
                            try:
                                page_dict = page.get_text("dict")
                                for block in page_dict.get("blocks", []):
                                    if "image" in block and (
                                        block.get("number") == xref
                                        or block.get("xref") == xref
                                        or block["image"] == xref
                                    ):
                                        image_rect = fitz.Rect(block["bbox"])
                                        print(
                                            f"Found image rect in page dict: {image_rect}"
                                        )
                                        break
                            except Exception as e:
                                print(f"Error searching in page dict: {str(e)}")

                        # Method 3: From raw dictionary
                        if not image_rect:
                            try:
                                raw_dict = page.get_text("rawdict")
                                for block in raw_dict.get("blocks", []):
                                    if block.get("type") == 1:  # Image block
                                        if block.get("xref") == xref:
                                            image_rect = fitz.Rect(block["bbox"])
                                            print(
                                                f"Found image rect in raw dict: {image_rect}"
                                            )
                                            break
                            except Exception as e:
                                print(f"Error searching in raw dict: {str(e)}")

                        # If we still don't have image_rect, create a default one
                        if not image_rect:
                            print(
                                f"Could not find image rect for xref {xref}, using default"
                            )
                            # Create a default rect at the top of the page
                            image_rect = fitz.Rect(
                                0, 0, page_rect.width / 2, page_rect.height / 2
                            )

                        # Get text below image
                        below_text = []
                        if image_rect:
                            # Define search area below image
                            search_area = fitz.Rect(
                                image_rect.x0 - 20,
                                image_rect.y1,
                                image_rect.x1 + 20,
                                min(page_rect.height, image_rect.y1 + 100),
                            )

                            # Get text in the search area
                            try:
                                text_below = page.get_text("text", clip=search_area)
                                if text_below.strip():
                                    below_text.append(text_below.strip())
                            except Exception as e:
                                print(f"Error getting text below image: {str(e)}")

                            # Try alternative method using blocks
                            try:
                                blocks = page.get_text("dict")["blocks"]
                                for block in blocks:
                                    if "lines" in block:
                                        block_rect = fitz.Rect(block["bbox"])
                                        if block_rect.intersects(search_area):
                                            text = " ".join(
                                                span["text"]
                                                for line in block["lines"]
                                                for span in line["spans"]
                                            )
                                            if (
                                                text.strip()
                                                and text.strip() not in below_text
                                            ):
                                                below_text.append(text.strip())
                            except Exception as e:
                                print(f"Error getting text from blocks: {str(e)}")

                        # Process image
                        try:
                            image_base64 = base64.b64encode(image_bytes).decode("utf-8")
                            img_info = {
                                "image": image_base64,
                                "image_bytes": image_bytes,
                                "page": page_num + 1,
                                "index": img_idx,
                                "bbox": list(image_rect),
                                "below_text": below_text,
                                "format": base_image["ext"],
                                "xref": xref,
                            }

                            # Get image size
                            try:
                                with Image.open(io.BytesIO(image_bytes)) as img_obj:
                                    img_info["width"] = img_obj.width
                                    img_info["height"] = img_obj.height
                            except Exception as e:
                                print(f"Error getting image size: {str(e)}")
                                img_info["width"] = None
                                img_info["height"] = None

                            # Get image analysis if needed
                            try:
                                img_info["analysis"] = self._analyze_image_with_llava(
                                    image_bytes
                                )
                            except Exception as e:
                                print(f"Error analyzing image: {str(e)}")
                                img_info["analysis"] = None

                            content["images"].append(img_info)
                            img_count += 1
                            print(f"Successfully processed image {img_count}")

                        except Exception as e:
                            print(f"Error processing image data: {str(e)}")

                    except Exception as img_error:
                        print(
                            f"Error processing image {img_idx} on page {page_num + 1}: {str(img_error)}"
                        )
                        continue

                print(f"Completed processing page {page_num + 1}")

            print(f"Total images processed: {img_count}")
            return content

        except Exception as e:
            print(f"Error extracting PDF content: {str(e)}")
            raise
        finally:
            if "doc" in locals():
                doc.close()

    def get_text_around_image(
        self, page: fitz.Page, rect: list, margin: int = 50
    ) -> str:
        """Get text around an image with specified margin"""
        try:
            # Create rectangle with margin
            image_rect = fitz.Rect(rect)
            search_rect = image_rect + (-margin, -margin, margin, margin)

            # Get text within expanded rectangle
            return page.get_text("text", clip=search_rect)
        except Exception as e:
            print(f"Error getting text around image: {str(e)}")
            return ""

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
            response = self.llm.invoke(messages)
            return response.content

        except Exception as e:
            print(f"Error analyzing image with LLaVA: {str(e)}")
            return "Failed to analyze image"

    def process_image_block(self, page: fitz.Page, block: dict) -> Optional[Dict]:
        """Process a single image block from PDF"""
        try:
            if "image" not in block:
                return None

            xref = block["image"]
            base_image = page.parent.extract_image(xref)

            if not base_image:
                return None

            # Get image data
            image_bytes = base_image["image"]
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")

            # Get surrounding text
            surrounding_text = self.get_text_around_image(page, block["bbox"])

            # Get image size
            with Image.open(io.BytesIO(image_bytes)) as img_obj:
                width, height = img_obj.size

            return {
                "image": image_base64,
                "image_bytes": image_bytes,
                "bbox": block["bbox"],
                "surrounding_text": surrounding_text,
                "width": width,
                "height": height,
                "format": base_image["ext"],
                "xref": xref,
            }

        except Exception as e:
            print(f"Error processing image block: {str(e)}")
            return None

    async def process_pdf(self, file_path: str, filename: str = None) -> Dict:
        """Process PDF with improved image and text handling"""
        try:
            pdf_content = self.extract_pdf_content(file_path)
            text_chunks = self.splitter.split_documents(pdf_content["text"])

            # Store text chunks in batches
            vector_store = Chroma(
                persist_directory=self.storage_dir,
                embedding_function=self.embedding,
                collection_name=settings.COLLECTION_NAME,
            )

            # Process text chunks in smaller batches
            batch_size = 50
            total_chunks_processed = 0
            for i in range(0, len(text_chunks), batch_size):
                batch = text_chunks[i : i + batch_size]
                try:
                    vector_store.add_documents(batch)
                    total_chunks_processed += len(batch)
                except Exception as chunk_error:
                    print(f"Error storing text chunk batch: {str(chunk_error)}")

            # Process and store images in batches
            saved_images = []
            batch_size = 10
            total_images_processed = 0

            for i in range(0, len(pdf_content["images"]), batch_size):
                image_batch = pdf_content["images"][i : i + batch_size]
                batch_documents = []
                batch_metadatas = []
                batch_ids = []

                for img_idx, img_data in enumerate(image_batch):
                    try:
                        # Save image
                        image_info = self.save_image(
                            img_data, img_data["page"], filename or file_path
                        )

                        # Prepare metadata
                        metadata = {
                            "page": str(img_data["page"]),
                            "source": str(filename or file_path),
                            "width": str(img_data.get("width", "")),
                            "height": str(img_data.get("height", "")),
                            "format": str(img_data.get("format", "")),
                            "type": "image",
                            "image_base64": img_data.get(
                                "image", ""
                            ),  # Save base64 in metadata
                            "analysis": str(img_data.get("analysis", "")),
                            "bbox": str(img_data.get("bbox", "")),
                            "xref": str(img_data.get("xref", "")),
                        }

                        # Handle below text
                        if isinstance(img_data.get("below_text"), list):
                            metadata["below_text"] = " ".join(img_data["below_text"])
                        elif isinstance(img_data.get("below_text"), str):
                            metadata["below_text"] = img_data["below_text"]
                        else:
                            metadata["below_text"] = ""

                        unique_id = f"img_{img_data['page']}_{i + img_idx}"

                        batch_documents.append(image_info["filename"])
                        batch_metadatas.append(metadata)
                        batch_ids.append(unique_id)

                        saved_images.append(image_info)
                        total_images_processed += 1

                    except Exception as e:
                        print(f"Error processing image in batch: {str(e)}")
                        continue

                # Store batch in ChromaDB
                if batch_documents:
                    try:
                        self.image_collection.add(
                            documents=batch_documents,
                            metadatas=batch_metadatas,
                            ids=batch_ids,
                        )
                    except Exception as chroma_error:
                        print(
                            f"Error storing image batch in ChromaDB: {str(chroma_error)}"
                        )

            vector_store.persist()

            return {
                "status": "success",
                "text_pages": len(pdf_content["text"]),
                "text_chunks": total_chunks_processed,
                "images_processed": total_images_processed,
                "message": f"Successfully processed {len(pdf_content['text'])} pages, {total_chunks_processed} text chunks, and {total_images_processed} images.",
            }

        except Exception as e:
            print(f"Error in process_pdf: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "message": "Failed to process PDF document.",
            }

    def save_image(self, img_data: Dict, page: int, source: str) -> Dict:
        """Save image and return metadata with enhanced error handling"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"page_{page}_{timestamp}.{img_data['format']}"
            file_path = os.path.join(self.image_dir, filename)

            try:
                Image.open(io.BytesIO(img_data["image_bytes"])).save(file_path)
            except Exception as save_error:
                print(f"Error saving image with PIL: {str(save_error)}")
                with open(file_path, "wb") as f:
                    f.write(img_data["image_bytes"])

            image_info = {
                "filename": filename,
                "path": file_path,
                "page": str(page),
                "source": str(source),
                "size": {
                    "width": str(img_data.get("width", "")),
                    "height": str(img_data.get("height", "")),
                },
                "format": str(img_data.get("format", "")),
                "file_size": str(os.path.getsize(file_path)),
                "timestamp": timestamp,
            }

            if img_data.get("below_text"):
                if isinstance(img_data["below_text"], list):
                    image_info["below_text"] = " ".join(img_data["below_text"])
                else:
                    image_info["below_text"] = str(img_data["below_text"])

            if img_data.get("bbox"):
                image_info["bbox"] = str(img_data["bbox"])

            image_info["xref"] = str(img_data.get("xref", ""))

            return image_info

        except Exception as e:
            print(f"Error in save_image: {str(e)}")
            raise

    def query_pdf(self, query: str, max_images: int = 5) -> tuple[str, List[Dict]]:
        """Enhanced PDF querying with multiple relevant images"""
        try:
            # Get relevant text
            vector_store = Chroma(
                persist_directory=self.storage_dir,
                embedding_function=self.embedding,
                collection_name=settings.COLLECTION_NAME,
            )
            docs = vector_store.similarity_search(query, k=settings.MAX_RESULTS)

            if not docs:
                return "No relevant information found.", []

            # Get relevant pages
            relevant_pages = {str(doc.metadata["page"]) for doc in docs}

            # Find relevant images
            all_images = self.image_collection.get()
            relevant_images = []

            if all_images and all_images["documents"]:
                for i, img_meta in enumerate(all_images["metadatas"]):
                    relevance_data = self._calculate_image_relevance(
                        query=query,
                        img_metadata=img_meta,
                        relevant_pages=relevant_pages,
                        below_text=img_meta.get("below_text", ""),
                    )

                    if relevance_data["total_score"] > 0.4:  # Adjusted threshold
                        formatted_image = {
                            "filename": all_images["documents"][i],
                            "image": img_meta.get(
                                "image_base64", ""
                            ),  # Get base64 directly from metadata
                            "metadata": {
                                "page": img_meta.get("page"),
                                "width": img_meta.get("width"),
                                "height": img_meta.get("height"),
                                "format": img_meta.get("format"),
                                "analysis": img_meta.get("analysis"),
                                "below_text": img_meta.get("below_text", ""),
                                "relevance_score": relevance_data["total_score"],
                                "relevance_details": relevance_data["detailed_scores"],
                            },
                        }
                        relevant_images.append(formatted_image)

            # Sort images by relevance and get top N most relevant
            relevant_images.sort(
                key=lambda x: x["metadata"]["relevance_score"], reverse=True
            )
            most_relevant_images = relevant_images[:max_images]

            # Prepare context
            context_text = "\n\n".join(
                f"[Page {doc.metadata['page']}] {doc.page_content}" for doc in docs
            )

            # Prepare image analysis for all relevant images
            image_analyses = []
            for img in most_relevant_images:
                analysis = f"""
                    Relevant image found on page {img['metadata']['page']}:
                    Caption/Text below: {img['metadata'].get('below_text', 'No caption available')}
                    Analysis: {img['metadata'].get('analysis', 'No analysis available')}
                    Relevance score: {img['metadata']['relevance_score']}
                    """
                image_analyses.append(analysis)

            # Get final answer
            answer = self.text_llm.invoke(
                self.qa_prompt.format(context=context_text, question=query)
            ).content

            return answer, most_relevant_images

        except Exception as e:
            print(f"Error in query_pdf: {str(e)}")
            raise

    def _calculate_image_relevance(
        self, query: str, img_metadata: Dict, relevant_pages: set, below_text: str
    ) -> Dict:
        """Calculate image relevance score with detailed scoring"""
        scores = {}
        query_terms = query.lower().split()

        # # 1. Page relevance (20%)
        # page_score = 0.2 if str(img_metadata.get("page", "")) in relevant_pages else 0
        # scores["page_relevance"] = page_score

        # # 2. Below text relevance (30%)
        # below_text = str(below_text).lower()
        # below_text_matches = sum(term in below_text for term in query_terms)
        # below_text_score = (
        #     0.3 * (below_text_matches / len(query_terms)) if query_terms else 0
        # )
        # scores["below_text_relevance"] = below_text_score

        # # 3. Image analysis relevance (30%)
        # analysis = str(img_metadata.get("analysis", "")).lower()
        # analysis_matches = sum(term in analysis for term in query_terms)
        # analysis_score = (
        #     0.3 * (analysis_matches / len(query_terms)) if query_terms else 0
        # )
        # scores["analysis_relevance"] = analysis_score

        # # 4. Caption/text match exactness (20%)
        # exact_phrase_matches = 0
        # if below_text:
        #     query_bigrams = set(zip(query_terms, query_terms[1:]))
        #     text_terms = below_text.split()
        #     text_bigrams = set(zip(text_terms, text_terms[1:]))
        #     bigram_matches = len(query_bigrams & text_bigrams)
        #     if bigram_matches > 0:
        #         exact_phrase_matches = 0.2 * (
        #             bigram_matches / len(query_bigrams) if query_bigrams else 0
        #         )
        # scores["exact_match_relevance"] = exact_phrase_matches

        # 1. Page relevance (30%)
        page_score = 0.3 if str(img_metadata.get("page", "")) in relevant_pages else 0
        scores["page_relevance"] = page_score

        # 2. Image analysis relevance (40%)
        analysis = str(img_metadata.get("analysis", "")).lower()
        analysis_matches = sum(term in analysis for term in query_terms)
        analysis_score = (
            0.4 * (analysis_matches / len(query_terms)) if query_terms else 0
        )
        scores["analysis_relevance"] = analysis_score

        # 3. Caption/text match exactness (30%)
        exact_phrase_matches = 0
        text_context = str(img_metadata.get("text_context", "")).lower()
        if text_context:
            query_bigrams = set(zip(query_terms, query_terms[1:]))
            text_terms = text_context.split()
            text_bigrams = set(zip(text_terms, text_terms[1:]))
            bigram_matches = len(query_bigrams & text_bigrams)
            if bigram_matches > 0:
                exact_phrase_matches = 0.3 * (
                    bigram_matches / len(query_bigrams) if query_bigrams else 0
                )
        scores["exact_match_relevance"] = exact_phrase_matches

        # Calculate total score
        total_score = sum(scores.values())

        # Normalize total score to 0-1 range
        total_score = min(1.0, total_score)

        return {"total_score": total_score, "detailed_scores": scores}

    # Save image to folder source local
    # def save_image(self, img_data: Dict, page: int, source: str) -> Dict:
    #     """Save image and return metadata with base64"""
    #     try:
    #         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #         filename = f"page_{page}_{timestamp}.{img_data['format']}"
    #         file_path = os.path.join(self.image_dir, filename)

    #         Image.open(io.BytesIO(img_data["image_bytes"])).save(file_path)

    #         return {
    #             "filename": filename,
    #             "path": file_path,
    #             "page": page,
    #             "source": source,
    #             "size": {"width": img_data["width"], "height": img_data["height"]},
    #             "format": img_data["format"],
    #             "file_size": os.path.getsize(file_path),
    #             "timestamp": timestamp,
    #             "image_base64": img_data.get("image", ""),
    #         }
    #     except Exception as e:
    #         print(f"Error saving image: {str(e)}")
    #         raise


# Initialize service
llm_manager = LLMService()
