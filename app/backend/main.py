from fastapi import FastAPI
from pydantic import BaseModel # For request body validation

app = FastAPI()

class MessagePayload(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {"message": "Hello from Simple FastAPI Backend! Phase 1 is A-OK!"}

@app.post("/echo/")
async def echo_message(payload: MessagePayload):
    return {"you_sent": payload.text, "backend_echoes": f"Backend received: {payload.text}"}