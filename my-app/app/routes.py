from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from .models import llm_manager
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

@app.post("/pdf")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        if not file.filename.lower().endswith('.pdf'):
            return JSONResponse(
                status_code=400,
                content={"error": "PDF files only"}
            )
            
        file_path = settings.PDF_DIR / file.filename
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
            
        doc_count, chunk_count = await llm_manager.process_pdf(
            str(file_path),
            file.filename
        )
        
        return {
            "status": "Success",
            "filename": file.filename,
            "documents": doc_count,
            "chunks": chunk_count
        }
        
    except Exception as e:
        print(f"Error in /pdf: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.post("/ask_pdf")
async def query_pdf(request: Request):
    try:
        data = await request.json()
        query = data.get("query")
        
        if not query:
            return JSONResponse(
                status_code=400,
                content={"error": "Query required"}
            )
            
        answer, sources = llm_manager.query_pdf(query)
        return {
            "answer": answer,
            "sources": sources
        }
        
    except Exception as e:
        print(f"Error in /ask_pdf: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

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