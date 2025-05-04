import os
import re
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
import uuid
import aiohttp
import time
import base64
import PyPDF2
from docx import Document
import pandas as pd
import odf
from odf import text, teletype
import pyth
import logging


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# first api key from openrouter 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
# second api key from openrouter  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
# third api key from openrouter  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
# API Configuration
API_KEY = "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"
BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

# Models
DEFAULT_MODEL = "meta-llama/llama-4-maverick:free"
THINK_MODEL = "google/gemini-2.0-flash-thinking-exp:free"
MULTIMODAL_MODEL = "qwen/qwen2.5-vl-72b-instruct:free"
YUPPI_MODEL = "yuppi-ai-model:free"  # Placeholder for Yuppi.AI model

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}


app = FastAPI(title="Veronica AI Assistant")

current_dir = os.path.dirname(os.path.abspath(__file__))
app.mount("/static", StaticFiles(directory=os.path.join(current_dir, "static")), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

chat_histories = {}

class Message(BaseModel):
    role: str
    content: str

class ChatSession(BaseModel):
    chat_id: str
    messages: List[Message]

class ChatRequest(BaseModel):
    query: str
    chat_id: Optional[str] = None
    model: Optional[str] = DEFAULT_MODEL

class ChatResponse(BaseModel):
    response: str
    chat_id: Optional[str]
    status: str
    processing_time: Optional[float] = None

@app.get("/")
async def read_root():
    return FileResponse(os.path.join(current_dir, "static", "index.html"))

@app.post("/veronica/chat", response_model=ChatSession)
async def create_chat():
    chat_id = str(uuid.uuid4())
    chat_histories[chat_id] = []
    return {"chat_id": chat_id, "messages": []}

@app.get("/veronica/chats", response_model=List[ChatSession])
async def get_all_chats():
    return [{"chat_id": chat_id, "messages": messages} for chat_id, messages in chat_histories.items()]

@app.get("/veronica/chat/{chat_id}", response_model=ChatSession)
async def get_chat_history(chat_id: str):
    if chat_id not in chat_histories:
        raise HTTPException(status_code=404, detail="Chat session not found")
    return {"chat_id": chat_id, "messages": chat_histories[chat_id]}

@app.delete("/veronica/chat/{chat_id}")
async def delete_chat(chat_id: str):
    if chat_id not in chat_histories:
        raise HTTPException(status_code=404, detail="Chat session not found")
    del chat_histories[chat_id]
    return {"status": "success"}

@app.delete("/veronica/chats")
async def clear_all_chats():
    chat_histories.clear()
    return {"status": "success"}

async def get_ai_response(messages: List[dict], model: str) -> str:
    payload = {
        "stream": False,
        "model": model,
        "messages": messages
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            logger.info(f"Sending request to OpenRouter with model: {model}")
            async with session.post(BASE_URL, json=payload, headers=HEADERS) as response:
                if response.status == 200:
                    response_json = await response.json()
                    full_response = response_json.get("choices", [{}])[0].get("message", {}).get("content", "")
                    clean_response = re.sub(r'<think>.*?</think>', '', full_response, flags=re.DOTALL).strip()
                    
                    if not clean_response:
                        clean_response = full_response.strip()
                    
                    if not clean_response:
                        raise HTTPException(status_code=500, detail="Empty response from API")
                    
                    logger.info(f"Received response: {clean_response[:100]}...")
                    return clean_response
                else:
                    error_text = await response.text()
                    logger.error(f"API Error: {error_text}")
                    # Fallback to default model if Think model fails
                    if model == THINK_MODEL:
                        logger.info("Falling back to default model due to Think model failure")
                        payload["model"] = DEFAULT_MODEL
                        async with session.post(BASE_URL, json=payload, headers=HEADERS) as fallback_response:
                            if fallback_response.status == 200:
                                fallback_json = await fallback_response.json()
                                fallback_response_text = fallback_json.get("choices", [{}])[0].get("message", {}).get("content", "")
                                return fallback_response_text
                            else:
                                fallback_error = await fallback_response.text()
                                logger.error(f"Fallback API Error: {fallback_error}")
                                raise HTTPException(status_code=response.status, detail=f"API Error: {error_text}, Fallback Error: {fallback_error}")
                    raise HTTPException(status_code=response.status, detail=f"API Error: {error_text}")
    except Exception as e:
        logger.error(f"Server error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

@app.post("/veronica", response_model=ChatResponse)
async def ask_veronica(request: ChatRequest):
    try:
        query = request.query
        chat_id = request.chat_id
        model = request.model or DEFAULT_MODEL

        if not query:
            raise HTTPException(status_code=400, detail="Query is required")

        if not API_KEY:
            raise HTTPException(status_code=500, detail="API key not found")

        # Map frontend model names to backend models
        if model == "veronica":
            model = DEFAULT_MODEL
        elif model == "yuppi":
            model = YUPPI_MODEL
        elif model == THINK_MODEL:
            model = THINK_MODEL
        else:
            model = DEFAULT_MODEL

        messages = []
        if chat_id:
            if chat_id not in chat_histories:
                chat_histories[chat_id] = []
            messages = chat_histories[chat_id].copy()

        if not any(msg.get("role") == "system" for msg in messages):
            system_message = {
                "role": "system", 
                "content": """You are a helpful assistant. Format your responses using proper markdown:
                - Use **bold** for emphasis
                - Use *italics* for subtle emphasis
                - Use proper headings with # for titles
                - Use - or * for bullet points
                - Use 1. 2. 3. for numbered lists
                - Use `code` for inline code
                - Use ```language for code blocks
                - Use > for quotes
                - Use [text](url) for links
                
                Provide clear and concise answers without additional tags or metadata."""
            }
            messages.insert(0, system_message)
        
        user_message = {"role": "user", "content": query}
        messages.append(user_message)
        
        start_time = time.time()
        response_text = await get_ai_response(messages, model)
        processing_time = time.time() - start_time if model == THINK_MODEL else None
        
        assistant_message = {"role": "assistant", "content": response_text}
        
        if chat_id:
            chat_histories[chat_id].append(user_message)
            chat_histories[chat_id].append(assistant_message)
        
        return {
            "response": response_text,
            "chat_id": chat_id,
            "status": "success",
            "processing_time": processing_time
        }

    except HTTPException as http_exc:
        raise
    except Exception as e:
        logger.error(f"Server error in ask_veronica: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

@app.post("/veronica/chat-with-document", response_model=ChatResponse)
async def chat_with_document(
    file: UploadFile = File(...),
    query: Optional[str] = Form(None),
    chat_id: Optional[str] = Form(None),
    model: Optional[str] = Form(DEFAULT_MODEL)
):
    try:
        # Map frontend model names to backend models
        if model == "veronica":
            model = DEFAULT_MODEL
        elif model == "yuppi":
            model = YUPPI_MODEL
        elif model == THINK_MODEL:
            model = THINK_MODEL
        else:
            model = DEFAULT_MODEL

        # Extract text from the file
        content = ""
        filename = file.filename.lower()
        
        # Simple text-based files
        if filename.endswith((
            '.txt', '.html', '.css', '.js', '.json', '.xml', '.yaml', '.yml', '.md',
            '.py', '.java', '.cpp', '.c', '.cs', '.sql', '.sh', '.bat', '.ts', '.jsx',
            '.tsx', '.php', '.log', '.ini', '.tex', '.bib'
        )):
            content = await file.read()
            content = content.decode('utf-8', errors='ignore')
        
        # PDF files
        elif filename.endswith('.pdf'):
            pdf_reader = PyPDF2.PdfReader(file.file)
            content = ""
            for page in pdf_reader.pages:
                content += page.extract_text() + "\n"
        
        # Microsoft Word (.docx)
        elif filename.endswith('.docx'):
            doc = Document(file.file)
            content = "\n".join([para.text for para in doc.paragraphs])
        
        # OpenDocument Text (.odt)
        elif filename.endswith('.odt'):
            doc = odf.opendocument.load(file.file)
            content = ""
            for element in doc.getElementsByType(text.P):
                content += teletype.extractText(element) + "\n"
        
        # Excel (.xlsx)
        elif filename.endswith('.xlsx'):
            df = pd.read_excel(file.file)
            content = df.to_string()
        
        # OpenDocument Spreadsheet (.ods)
        elif filename.endswith('.ods'):
            doc = odf.opendocument.load(file.file)
            content = ""
            for element in doc.getElementsByType(text.P):
                content += teletype.extractText(element) + "\n"
        
        # RTF files
        elif filename.endswith('.rtf'):
            content = await file.read()
            content = pyth.rtf_to_text(content.decode('utf-8', errors='ignore'))
        
        # CSV files
        elif filename.endswith('.csv'):
            df = pd.read_csv(file.file)
            content = df.to_string()
        
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")

        if not content.strip():
            raise HTTPException(status_code=400, detail="Не удалось извлечь текст из файла")

        # Formulate the query
        full_query = f"Содержимое документа:\n{content}\n\n{query if query else 'Анализируй документ.'}"
        
        # Send to OpenRouter
        messages = []
        if chat_id:
            if chat_id not in chat_histories:
                chat_histories[chat_id] = []
            messages = chat_histories[chat_id].copy()

        if not any(msg.get("role") == "system" for msg in messages):
            system_message = {
                "role": "system", 
                "content": """You are a helpful assistant. Format your responses using proper markdown:
                - Use **bold** for emphasis
                - Use *italics* for subtle emphasis
                - Use proper headings with # for titles
                - Use - or * for bullet points
                - Use 1. 2. 3. for numbered lists
                - Use `code` for inline code
                - Use ```language for code blocks
                - Use > for quotes
                - Use [text](url) for links
                
                Provide clear and concise answers without additional tags or metadata."""
            }
            messages.insert(0, system_message)

        user_message = {"role": "user", "content": full_query}
        messages.append(user_message)
        
        response_text = await get_ai_response(messages, model)

        # Save to chat history
        if chat_id:
            chat_histories[chat_id].append({"role": "user", "content": f"Документ загружен: {file.filename}\n\n{full_query}"})
            chat_histories[chat_id].append({"role": "assistant", "content": response_text})

        return {
            "response": response_text,
            "chat_id": chat_id,
            "status": "success"
        }

    except HTTPException as http_exc:
        raise
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

@app.post("/veronica/chat-with-image", response_model=ChatResponse)
async def chat_with_image(
    file: UploadFile = File(...),
    query: Optional[str] = Form(None),
    chat_id: Optional[str] = Form(None),
    model: Optional[str] = Form(None)
):
    try:
        # Use the default multimodal model if none is specified
        if not model:
            model = MULTIMODAL_MODEL
        elif model not in [MULTIMODAL_MODEL, "google/gemini-2.5-pro-exp-03-25:free"]:
            raise HTTPException(status_code=400, detail="Модель не поддерживает обработку изображений. Выберите мультимодальную модель (например, qwen/qwen2.5-vl-72b-instruct:free).")

        # Validate image format
        filename = file.filename.lower()
        if not filename.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp')):
            raise HTTPException(status_code=400, detail="Unsupported image format. Supported formats: .jpg, .jpeg, .png, .gif, .bmp, .webp")

        # Encode image to base64
        image_content = await file.read()
        base64_image = base64.b64encode(image_content).decode('utf-8')
        content_type = file.content_type or "image/jpeg"

        # Formulate the request
        message_content = [
            {"type": "text", "text": query or "Опиши изображение."},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{content_type};base64,{base64_image}"
                }
            }
        ]
        
        messages = []
        if chat_id:
            if chat_id not in chat_histories:
                chat_histories[chat_id] = []
            messages = chat_histories[chat_id].copy()

        if not any(msg.get("role") == "system" for msg in messages):
            system_message = {
                "role": "system", 
                "content": """You are a helpful assistant. Format your responses using proper markdown:
                - Use **bold** for emphasis
                - Use *italics* for subtle emphasis
                - Use proper headings with # for titles
                - Use - or * for bullet points
                - Use 1. 2. 3. for numbered lists
                - Use `code` for inline code
                - Use ```language for code blocks
                - Use > for quotes
                - Use [text](url) for links
                
                Provide clear and concise answers without additional tags or metadata."""
            }
            messages.insert(0, system_message)

        messages.append({"role": "user", "content": message_content})

        # Send to OpenRouter
        payload = {
            "stream": False,
            "model": model,
            "messages": messages
        }
        
        async with aiohttp.ClientSession() as session:
            logger.info(f"Sending image request to OpenRouter with model: {model}")
            async with session.post(BASE_URL, json=payload, headers=HEADERS) as response:
                if response.status == 200:
                    response_json = await response.json()
                    response_text = response_json.get("choices", [{}])[0].get("message", {}).get("content", "")
                    if not response_text:
                        raise HTTPException(status_code=500, detail="Empty response from API")
                else:
                    error_text = await response.text()
                    logger.error(f"Image API Error: {error_text}")
                    raise HTTPException(status_code=response.status, detail=f"API Error: {error_text}")

        # Save to chat history
        if chat_id:
            chat_histories[chat_id].append({"role": "user", "content": f"Изображение загружено: {file.filename}"})
            chat_histories[chat_id].append({"role": "assistant", "content": response_text})

        return {
            "response": response_text,
            "chat_id": chat_id,
            "status": "success"
        }

    except HTTPException as http_exc:
        raise
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.get("/veronica/models")
async def get_models():
    return {
        "default_models": [
            {"name": "Veronica", "id": "veronica"},
            {"name": "Yuppi.AI", "id": "yuppi"}
        ],
        "think_model": THINK_MODEL,
        "multimodal_model": MULTIMODAL_MODEL
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)