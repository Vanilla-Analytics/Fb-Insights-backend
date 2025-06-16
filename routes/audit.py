from fastapi import APIRouter
from fastapi import FastAPI
import requests
from services.deepseek_audit import generate_audit
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import traceback

router = APIRouter()

class AuditRequest(BaseModel):
    access_token: str
    page_id: str

@router.post("/audit")
async def get_audit(request: AuditRequest):
    try:
        print(f"🔄 Starting audit for page_id: {request.page_id}")
        
        # Validate input
        if not request.access_token or not request.page_id:
            return JSONResponse(
                content={"error": "access_token and page_id are required"}, 
                status_code=400
            )
        
        # Generate audit and return PDF
        pdf_response = await generate_audit(request.page_id, request.access_token)
        print("✅ Audit completed successfully")
        return pdf_response
        
    except ValueError as ve:
        error_msg = f"Validation error: {str(ve)}"
        print(f"❌ {error_msg}")
        return JSONResponse(content={"error": error_msg}, status_code=400)
    except Exception as e:
        error_msg = f"Internal server error: {str(e)}"
        print(f"❌ {error_msg}")
        print(f"❌ Traceback: {traceback.format_exc()}")
        return JSONResponse(content={"error": error_msg}, status_code=500)