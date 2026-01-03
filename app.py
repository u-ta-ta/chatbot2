from fastapi import FastAPI
from pydantic import BaseModel
import re

from chatbot import chatbot_response

app = FastAPI()

class ChatRequest(BaseModel):
    message: str

def format_response_with_link(text):
    url_pattern = r'http[s]?://[^\s]+'
    return re.sub(url_pattern, r'<a href="\g<0>" target="_blank">\g<0></a>', text)

@app.post("/chat")
def chat(req: ChatRequest):
    if not req.message:
        return {"response": "Vui lòng nhập câu hỏi!"}

    response = chatbot_response(req.message)
    return {"response": format_response_with_link(response)}
