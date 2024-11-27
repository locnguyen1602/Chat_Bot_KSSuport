import base64
from datetime import datetime
from fastapi import FastAPI, HTTPException, Request, File, Response, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
# from .models3 import llm_manager3
from .models_with_memory import llm_manager
from .config import settings

app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION
)

# Configure CORS
origins = [
    "http://localhost",
    "http://localhost:52677",
    "http://localhost:5000",
    "http://localhost:5001",
    "*"  # Allow all origins (remove in production)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/ai")
async def ai_query(request: Request):
    try:
        data = await request.json()
        query = data.get("query")

        if not query:
            return JSONResponse(
                status_code=400,
                content={"error": "Query required"}
            )

        response = llm_manager.get_answer(query)
        return {"answer": response}

    except Exception as e:
        print(f"Error in /ai: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


# Route handlers
@app.post("/pdf")
async def process_pdf_route(file: UploadFile = File(...)):
    try:
        # Lưu file tạm thời
        temp_file_path = f"temp_{file.filename}"
        with open(temp_file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Process PDF
        result = await llm_manager.process_pdf(temp_file_path, file.filename)

        # Xóa file tạm
        import os

        os.remove(temp_file_path)

        return JSONResponse(status_code=200, content=result)

    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/ask_pdf")
async def query_pdf(request: Request):
    """Endpoint to query PDF and get text response only"""
    try:
        # Get request data
        data = await request.json()
        query = data.get("query")

        # Validate query
        if not query:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": "Query required",
                },
            )

        # Get response from LLM
        answer = llm_manager.query_pdf(query)

        # Format response
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "query": query,
                "answer": answer,
                "timestamp": datetime.now().isoformat(),
            },
        )

    except ValueError as ve:
        # Handle validation errors
        return JSONResponse(
            status_code=400, content={"status": "error", "message": str(ve)}
        )

    except Exception as e:
        # Log error
        print(f"Error in /ask_pdf: {str(e)}")

        # Return error response
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": "An error occurred while processing your request",
                "error": str(e),
            },
        )


# Peding with version do
@app.post("/ask_pdf_image")
async def query_pdf_image(request: Request):
    try:
        data = await request.json()
        query = data.get("query")
        max_images = data.get("max_images", 5)  # Optional parameter with default value

        if not query:
            return JSONResponse(
                status_code=400,
                content={"error": "Query required"}
            )

        answer, images = llm_manager.query_pdf(query)

        # Format all relevant images
        formatted_images = []
        if images and len(images) > 0:
            for image in images[:max_images]:  # Limit to max_images
                formatted_image = {
                    "filename": image["filename"],
                    "image": image["image"],  # Base64 string
                    "metadata": {
                        "format": image["metadata"].get("format"),
                        "page": image["metadata"].get("page"),
                        "width": image["metadata"].get("width"),
                        "height": image["metadata"].get("height"),
                        "relevance_score": image["metadata"].get("relevance_score", 0),
                        "below_text": image["metadata"].get("below_text", ""),
                        "analysis": image["metadata"].get("analysis", ""),
                    },
                }
                formatted_images.append(formatted_image)

        return JSONResponse(
            status_code=200,
            content={
                "answer": answer,
                "images": formatted_images,
                "total_images": len(formatted_images),
                "status": "success",
                "query": query,
            },
        )

    except Exception as e:
        print(f"Error in /ask_pdf: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "status": "error",
                "message": "An error occurred while processing your request",
            },
        )


@app.post("/clear-history")
async def clear_history():
    """Clear chat history"""
    try:
        llm_manager.clear_chat_history()
        return {"status": "success", "message": "Chat history cleared"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/images")
async def get_all_images():
    try:
        images = llm_manager.get_saved_images()
        if not images:
            return JSONResponse(
                status_code=404, content={"error": "No images found", "images": []}
            )

        return {"total": len(images), "images": images}

    except Exception as e:
        print(f"Error in /images: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/storage/images")
async def get_all_storage_images():
    try:
        result = llm_manager.get_storage_images()

        if result["total"] == 0:
            return JSONResponse(
                status_code=404,
                content={
                    "error": "No images found in storage",
                    "total": 0,
                    "images": [],
                },
            )

        return {
            "status": "success",
            "total": result["total"],
            "images": result["images"],
        }

    except Exception as e:
        print(f"Error in /storage/images: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})


# Add OPTIONS handler for CORS preflight requests
@app.options("/{full_path:path}")
async def options_handler(request: Request, full_path: str):
    return JSONResponse(
        content={},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "*",
        },
    )
