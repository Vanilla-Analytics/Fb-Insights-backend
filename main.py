from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from routes.audit import router as audit_router
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("FRONTEND_URL", "*")],
    #allow_origins=["https://fb-insights-production-f726.up.railway.app/"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



app.include_router(audit_router, prefix="/api")

@app.get("/")
async def root():
    return {"message": "Backend is running"}
