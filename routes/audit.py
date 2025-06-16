from fastapi import APIRouter
from fastapi import FastAPI
import requests
from services.deepseek_audit import generate_audit
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from services.deepseek_audit import generate_audit_report, generate_pdf_report

router = APIRouter()

class AuditRequest(BaseModel):
    access_token: str
    page_id: str



# @router.post("/audit")
# async def get_audit(request: AuditRequest):
#     try:
#         audit_report = await generate_audit(request.page_id, request.access_token)
#         return JSONResponse(content={"report": audit_report})
#     except Exception as e:
#         return JSONResponse(content={"error": str(e)}, status_code=500)

@router.post("/audit")
async def get_audit(request: AuditRequest):
    try:
        audit_report = await generate_audit_report(request.page_id, request.access_token)
        return generate_pdf_report(audit_report)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

