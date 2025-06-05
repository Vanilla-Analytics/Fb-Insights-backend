from fastapi import APIRouter, Request
from services.deepseek_audit import generate_audit
from pydantic import BaseModel

router = APIRouter()

class AuditRequest(BaseModel):
    access_token: str
    page_id: str

@router.post("/audit")
async def get_audit(request: AuditRequest):
    audit_report = await generate_audit(request.access_token, request.page_id)
    return {"report": audit_report}
