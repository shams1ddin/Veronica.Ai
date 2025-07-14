import os
import re
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
import uuid
import aiohttp
import time
import base64
import PyPDF2
from docx import Document
import pandas as pd
from odf import text, teletype
from striprtf.striprtf import rtf_to_text
import logging
import tempfile
import mimetypes
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API Keys Configuration
API_KEYS = [
    {"id": "first api", "key": "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"},
    {"id": "second api", "key": "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"},
    {"id": "third api", "key": "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"},
    {"id": "fourth api", "key": "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"}
]

# Google Custom Search API Configuration
GOOGLE_API_KEY = "AIzaSyBsT2LWAus5KJ2gCagkZSYORm8QA8IgMJs"
GOOGLE_SEARCH_ENGINE_ID = "531f88e98ea35413d"

# Hugging Face API Configuration
HUGGING_FACE_API_KEY = "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"
IMAGE_GEN_MODEL = "black-forest-labs/FLUX.1-schnell"

# Global variable to track current API key
current_api_key = API_KEYS[0]["key"]
current_api_key_id = API_KEYS[0]["id"]

BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

# Models
DEFAULT_MODEL = "deepseek/deepseek-chat-v3-0324:free"
THINK_MODEL = "google/gemini-2.0-flash-thinking-exp:free"
MULTIMODAL_MODEL = "qwen/qwen2.5-vl-72b-instruct:free"
YUPPI_MODEL = "deepseek/deepseek-chat-v3-0324:free"

def get_headers():
    return {
        "Authorization": f"Bearer {current_api_key}",
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
uploaded_files = {}

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

class FileChatRequest(BaseModel):
    file_url: str
    query: Optional[str] = None
    chat_id: Optional[str] = None
    model: Optional[str] = DEFAULT_MODEL

class MultipleFilesChatRequest(BaseModel):
    files: List[Dict[str, str]]
    query: Optional[str] = None
    chat_id: Optional[str] = None
    model: Optional[str] = MULTIMODAL_MODEL

class ChatResponse(BaseModel):
    response: str
    chat_id: Optional[str]
    status: str
    processing_time: Optional[float] = None
    api_key_changed: Optional[bool] = False
    error: Optional[str] = None

class UploadResponse(BaseModel):
    url: str
    filename: str

class APIKeyResponse(BaseModel):
    keys: List[Dict[str, str]]
    current_key_id: str

class APIKeySwitchRequest(BaseModel):
    key_id: str

class DeepSearchRequest(BaseModel):
    query: str
    chat_id: Optional[str] = None
    model: Optional[str] = DEFAULT_MODEL

class ImageGenerationRequest(BaseModel):
    prompt: str
    chat_id: Optional[str] = None
    model: Optional[str] = IMAGE_GEN_MODEL

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

@app.post("/veronica/upload-image", response_model=UploadResponse)
async def upload_image(file: UploadFile = File(...), chat_id: Optional[str] = Form(None)):
    filename = file.filename.lower()
    if not filename.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp')):
        raise HTTPException(status_code=400, detail="Unsupported image format")
    
    file_id = str(uuid.uuid4())
    content_type = file.content_type or "image/jpeg"
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_path = tmp_file.name
    
    uploaded_files[file_id] = {
        "path": tmp_path,
        "filename": filename,
        "content_type": content_type
    }
    
    file_url = f"/files/{file_id}"
    
    return {"url": file_url, "filename": filename}

@app.post("/veronica/upload-document", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...), chat_id: Optional[str] = Form(None)):
    filename = file.filename.lower()
    supported_extensions = (
        '.pdf', '.txt', '.docx', '.odt', '.xlsx', '.ods', '.rtf', '.csv',
        '.html', '.css', '.js', '.json', '.xml', '.yaml', '.yml', '.md',
        '.py', '.java', '.cpp', '.c', '.cs', '.sql', '.sh', '.bat', '.ts',
        '.jsx', '.tsx', '.php', '.log', '.ini', '.tex', '.bib'
    )
    if not filename.endswith(supported_extensions):
        raise HTTPException(status_code=400, detail="Unsupported document format")
    
    file_id = str(uuid.uuid4())
    content_type = file.content_type or "application/octet-stream"
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_path = tmp_file.name
    
    uploaded_files[file_id] = {
        "path": tmp_path,
        "filename": filename,
        "content_type": content_type
    }
    
    file_url = f"/files/{file_id}"
    
    return {"url": file_url, "filename": filename}

@app.get("/files/{file_id}")
async def get_file(file_id: str):
    if file_id not in uploaded_files:
        raise HTTPException(status_code=404, detail="File not found")
    
    file_info = uploaded_files[file_id]
    return FileResponse(
        path=file_info["path"],
        filename=file_info["filename"],
        media_type=file_info["content_type"]
    )

async def get_ai_response(messages: List[dict], model: str) -> tuple[str, bool]:
    global current_api_key, current_api_key_id
    api_key_changed = False
    current_key_index = next(i for i, key in enumerate(API_KEYS) if key["key"] == current_api_key)
    
    payload = {
        "stream": False,
        "model": model,
        "messages": messages
    }
    
    while True:
        try:
            async with aiohttp.ClientSession() as session:
                logger.info(f"Sending request to OpenRouter with model: {model}, key: {current_api_key_id}")
                async with session.post(BASE_URL, json=payload, headers=get_headers()) as response:
                    if response.status == 200:
                        response_json = await response.json()
                        full_response = response_json.get("choices", [{}])[0].get("message", {}).get("content", "")
                        clean_response = re.sub(r'<think>.*?</think>', '', full_response, flags=re.DOTALL).strip()
                        
                        if not clean_response:
                            clean_response = full_response.strip()
                        
                        if not clean_response:
                            raise HTTPException(status_code=500, detail="Empty response from API")
                        
                        logger.info(f"Received response: {clean_response[:100]}...")
                        return clean_response, api_key_changed
                    
                    elif response.status == 429 or response.status >= 500:  # Rate limit or server error
                        error_text = await response.text()
                        logger.warning(f"API Error with key {current_api_key_id}: {error_text}")
                        
                        # Try next API key
                        current_key_index = (current_key_index + 1) % len(API_KEYS)
                        current_api_key = API_KEYS[current_key_index]["key"]
                        current_api_key_id = API_KEYS[current_key_index]["id"]
                        api_key_changed = True
                        logger.info(f"Switched to API key: {current_api_key_id}")
                        
                        if current_key_index == 0:  # We've tried all keys
                            raise HTTPException(status_code=429, detail="All API keys have reached their limits")
                        continue
                    
                    else:
                        error_text = await response.text()
                        logger.error(f"API Error: {error_text}")
                        if model == THINK_MODEL:
                            logger.info("Falling back to default model")
                            payload["model"] = DEFAULT_MODEL
                            async with session.post(BASE_URL, json=payload, headers=get_headers()) as fallback_response:
                                if fallback_response.status == 200:
                                    fallback_json = await fallback_response.json()
                                    fallback_response_text = fallback_json.get("choices", [{}])[0].get("message", {}).get("content", "")
                                    return fallback_response_text, api_key_changed
                                else:
                                    fallback_error = await fallback_response.text()
                                    logger.error(f"Fallback API Error: {fallback_error}")
                                    raise HTTPException(status_code=response.status, detail=f"API Error: {error_text}, Fallback Error: {fallback_error}")
                        raise HTTPException(status_code=response.status, detail=f"API Error: {error_text}")
        
        except Exception as e:
            logger.error(f"Server error: {str(e)}")
            if isinstance(e, HTTPException) and e.status_code == 429:
                # Try next API key
                current_key_index = (current_key_index + 1) % len(API_KEYS)
                current_api_key = API_KEYS[current_key_index]["key"]
                current_api_key_id = API_KEYS[current_key_index]["id"]
                api_key_changed = True
                logger.info(f"Switched to API key: {current_api_key_id}")
                
                if current_key_index == 0:  # We've tried all keys
                    raise HTTPException(status_code=429, detail="All API keys have reached their limits")
                continue
            raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

@app.post("/veronica", response_model=ChatResponse)
async def ask_veronica(request: ChatRequest):
    try:
        query = request.query
        chat_id = request.chat_id
        model = request.model or DEFAULT_MODEL

        if not query:
            raise HTTPException(status_code=400, detail="Query is required")

        if not current_api_key:
            raise HTTPException(status_code=500, detail="API key not found")

        model_map = {
            "veronica": DEFAULT_MODEL,
            "yuppi": YUPPI_MODEL,
            THINK_MODEL: THINK_MODEL,
            MULTIMODAL_MODEL: MULTIMODAL_MODEL
        }
        model = model_map.get(model, DEFAULT_MODEL)

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
        response_text, api_key_changed = await get_ai_response(messages, model)
        processing_time = time.time() - start_time if model == THINK_MODEL else None
        
        assistant_message = {"role": "assistant", "content": response_text}
        
        if chat_id:
            chat_histories[chat_id].append(user_message)
            chat_histories[chat_id].append(assistant_message)
        
        return {
            "response": response_text,
            "chat_id": chat_id,
            "status": "success",
            "processing_time": processing_time,
            "api_key_changed": api_key_changed
        }

    except HTTPException as http_exc:
        if http_exc.status_code == 429:
            return JSONResponse(
                status_code=429,
                content={
                    "response": "API rate limit exceeded. Please switch API key.",
                    "status": "error",
                    "error": str(http_exc.detail),
                    "api_key_changed": True
                }
            )
        raise
    except Exception as e:
        logger.error(f"Server error in ask_veronica: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

@app.post("/veronica/chat-with-document", response_model=ChatResponse)
async def chat_with_document(request: FileChatRequest):
    try:
        file_url = request.file_url
        query = request.query
        chat_id = request.chat_id
        model = request.model or DEFAULT_MODEL

        model_map = {
            "veronica": DEFAULT_MODEL,
            "yuppi": YUPPI_MODEL,
            THINK_MODEL: THINK_MODEL
        }
        model = model_map.get(model, DEFAULT_MODEL)

        file_id = file_url.split('/')[-1]
        if file_id not in uploaded_files:
            raise HTTPException(status_code=404, detail="File not found")

        file_info = uploaded_files[file_id]
        filename = file_info["filename"]
        file_path = file_info["path"]

        content = ""
        if filename.endswith((
            '.txt', '.html', '.css', '.js', '.json', '.xml', '.yaml', '.yml', '.md',
            '.py', '.java', '.cpp', '.c', '.cs', '.sql', '.sh', '.bat', '.ts', '.jsx',
            '.tsx', '.php', '.log', '.ini', '.tex', '.bib'
        )):
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        
        elif filename.endswith('.pdf'):
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for page in pdf_reader.pages:
                    content += page.extract_text() + "\n"
        
        elif filename.endswith('.docx'):
            doc = Document(file_path)
            content = "\n".join([para.text for para in doc.paragraphs])
        
        elif filename.endswith('.odt'):
            doc = odf.opendocument.load(file_path)
            for element in doc.getElementsByType(text.P):
                content += teletype.extractText(element) + "\n"
        
        elif filename.endswith('.xlsx'):
            df = pd.read_excel(file_path)
            content = df.to_string()
        
        elif filename.endswith('.ods'):
            doc = odf.opendocument.load(file_path)
            for element in doc.getElementsByType(text.P):
                content += teletype.extractText(element) + "\n"
        
        elif filename.endswith('.rtf'):
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = rtf_to_text(f.read())
        
        elif filename.endswith('.csv'):
            df = pd.read_csv(file_path)
            content = df.to_string()
        
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")

        if not content.strip():
            raise HTTPException(status_code=400, detail="Не удалось извлечь текст из файла")

        full_query = f"Содержимое документа:\n{content}\n\n{query if query else 'Анализируй документ.'}"
        
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
        
        response_text, api_key_changed = await get_ai_response(messages, model)

        if chat_id:
            chat_histories[chat_id].append({"role": "user", "content": f"Документ загружен: {filename}\n\n{full_query}"})
            chat_histories[chat_id].append({"role": "assistant", "content": response_text})

        return {
            "response": response_text,
            "chat_id": chat_id,
            "status": "success",
            "api_key_changed": api_key_changed
        }

    except HTTPException as http_exc:
        if http_exc.status_code == 429:
            return JSONResponse(
                status_code=429,
                content={
                    "response": "API rate limit exceeded. Please switch API key.",
                    "status": "error",
                    "error": str(http_exc.detail),
                    "api_key_changed": True
                }
            )
        raise
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

@app.post("/veronica/chat-with-image", response_model=ChatResponse)
async def chat_with_image(request: FileChatRequest):
    try:
        file_url = request.file_url
        query = request.query
        chat_id = request.chat_id
        model = request.model or MULTIMODAL_MODEL

        if model not in [MULTIMODAL_MODEL, "google/gemini-2.5-pro-exp-03-25:free"]:
            raise HTTPException(status_code=400, detail="Модель не поддерживает обработку изображений")

        file_id = file_url.split('/')[-1]
        if file_id not in uploaded_files:
            raise HTTPException(status_code=404, detail="File not found")

        file_info = uploaded_files[file_id]
        filename = file_info["filename"]
        file_path = file_info["path"]
        content_type = file_info["content_type"]

        with open(file_path, 'rb') as f:
            image_content = f.read()
        base64_image = base64.b64encode(image_content).decode('utf-8')

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

        response_text, api_key_changed = await get_ai_response(messages, model)

        if chat_id:
            chat_histories[chat_id].append({"role": "user", "content": f"Изображение загружено: {filename}"})
            chat_histories[chat_id].append({"role": "assistant", "content": response_text})

        return {
            "response": response_text,
            "chat_id": chat_id,
            "status": "success",
            "api_key_changed": api_key_changed
        }

    except HTTPException as http_exc:
        if http_exc.status_code == 429:
            return JSONResponse(
                status_code=429,
                content={
                    "response": "API rate limit exceeded. Please switch API key.",
                    "status": "error",
                    "error": str(http_exc.detail),
                    "api_key_changed": True
                }
            )
        raise
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/veronica/chat-with-multiple-images", response_model=ChatResponse)
async def chat_with_multiple_images(request: MultipleFilesChatRequest):
    try:
        files = request.files
        query = request.query
        chat_id = request.chat_id
        model = request.model or MULTIMODAL_MODEL

        if model not in [MULTIMODAL_MODEL, "google/gemini-2.5-pro-exp-03-25:free"]:
            raise HTTPException(status_code=400, detail="Модель не поддерживает обработку изображений")

        if not files:
            raise HTTPException(status_code=400, detail="At least one file is required")

        message_content = [{"type": "text", "text": query or "Проанализируй все эти изображения вместе и опиши их."}]
        
        for file_info in files:
            file_url = file_info.get("file_url")
            if not file_url:
                continue
                
            file_id = file_url.split('/')[-1]
            if file_id not in uploaded_files:
                continue

            file_data = uploaded_files[file_id]
            with open(file_data["path"], 'rb') as f:
                image_content = f.read()
            base64_image = base64.b64encode(image_content).decode('utf-8')
            
            message_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{file_data['content_type']};base64,{base64_image}"
                }
            })
        
        messages = []
        if chat_id:
            if chat_id not in chat_histories:
                chat_histories[chat_id] = []
            messages = chat_histories[chat_id].copy()

        if not any(msg.get("role") == "system" for msg in messages):
            system_message = {
                "role": "system", 
                "content": """Ты - помощник, который анализирует несколько изображений одновременно. 
                При ответе:
                1. Сначала проанализируй все изображения вместе
                2. Найди связи между изображениями
                3. Дай общий контекстный ответ, учитывающий все изображения
                4. Если есть особенности или детали в отдельных изображениях - укажи их
                
                Используй markdown для форматирования:
                - **жирный** для выделения
                - *курсив* для подчеркивания
                - Заголовки с #
                - Списки с - или *
                - Нумерованные списки с 1. 2. 3.
                
                Давай четкие и понятные ответы."""
            }
            messages.insert(0, system_message)

        messages.append({"role": "user", "content": message_content})

        response_text, api_key_changed = await get_ai_response(messages, model)

        if chat_id:
            chat_histories[chat_id].append({
                "role": "user", 
                "content": f"Загружено несколько изображений для анализа: {', '.join(f['filename'] for f in [uploaded_files[url.split('/')[-1]] for url in [f['file_url'] for f in files]])}"
            })
            chat_histories[chat_id].append({"role": "assistant", "content": response_text})

        return {
            "response": response_text,
            "chat_id": chat_id,
            "status": "success",
            "api_key_changed": api_key_changed
        }

    except HTTPException as http_exc:
        if http_exc.status_code == 429:
            return JSONResponse(
                status_code=429,
                content={
                    "response": "API rate limit exceeded. Please switch API key.",
                    "status": "error",
                    "error": str(http_exc.detail),
                    "api_key_changed": True
                }
            )
        raise
    except Exception as e:
        logger.error(f"Error processing multiple images: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing multiple images: {str(e)}")

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

@app.get("/veronica/api-keys", response_model=APIKeyResponse)
async def get_api_keys():
    return {
        "keys": [{"id": key["id"], "name": key["id"]} for key in API_KEYS],
        "current_key_id": current_api_key_id
    }

@app.post("/veronica/switch-api-key")
async def switch_api_key(request: APIKeySwitchRequest):
    global current_api_key, current_api_key_id
    key_id = request.key_id
    key_info = next((key for key in API_KEYS if key["id"] == key_id), None)
    if not key_info:
        raise HTTPException(status_code=404, detail="API key not found")
    
    current_api_key = key_info["key"]
    current_api_key_id = key_info["id"]
    logger.info(f"Manually switched to API key: {current_api_key_id}")
    
    return {"status": "success", "current_key_id": current_api_key_id}

@app.post("/veronica/deepsearch", response_model=ChatResponse)
async def deep_search(request: DeepSearchRequest):
    try:
        query = request.query
        chat_id = request.chat_id
        model = request.model or DEFAULT_MODEL
        
        # Проверяем, что модель поддерживает DeepSearch
        model_map = {
            "veronica": DEFAULT_MODEL,
            "yuppi": YUPPI_MODEL,
            THINK_MODEL: THINK_MODEL,
            MULTIMODAL_MODEL: MULTIMODAL_MODEL
        }
        model = model_map.get(model, DEFAULT_MODEL)

        if not query:
            raise HTTPException(status_code=400, detail="Query is required")

        # Выполняем поиск через Google Custom Search API
        search_url = f"https://www.googleapis.com/customsearch/v1?key={GOOGLE_API_KEY}&cx={GOOGLE_SEARCH_ENGINE_ID}&q={query}"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(search_url) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Google Search API Error: {error_text}")
                    raise HTTPException(status_code=response.status, detail=f"Google Search API Error: {error_text}")
                
                search_results = await response.json()

        # Формируем результаты поиска для анализа ИИ
        search_content = ""
        
        if "items" in search_results and len(search_results["items"]) > 0:
            search_content += "# Результаты веб поиска\n\n"
            
            for i, item in enumerate(search_results["items"], 1):
                title = item.get("title", "Без заголовка")
                link = item.get("link", "#")
                snippet = item.get("snippet", "Нет описания")
                
                search_content += f"## {i}. [{title}]({link})\n"
                search_content += f"{snippet}\n\n"
        else:
            search_content += "По вашему запросу ничего не найдено.\n"

        # Создаем сообщения для отправки в модель ИИ
        messages = []
        system_message = {
            "role": "system", 
            "content": """Ты - помощник с функцией DeepSearch, который анализирует результаты веб-поиска и дает на их основе полезные ответы.
            При ответе:
            1. Проанализируй все результаты поиска
            2. Выдели ключевую информацию по запросу пользователя
            3. Структурируй ответ логически
            4. Если информации недостаточно, укажи это
            5. Всегда указывай источники информации
            
            Используй markdown для форматирования:
            - **жирный** для выделения
            - *курсив* для подчеркивания
            - Заголовки с #
            - Списки с - или *
            - Нумерованные списки с 1. 2. 3.
            
            Давай четкие и понятные ответы."""
        }
        
        messages.append(system_message)
        messages.append({"role": "user", "content": f"Запрос пользователя: {query}\n\nРезультаты поиска:\n{search_content}"})
        
        # Получаем ответ от модели ИИ
        start_time = time.time()
        response_text, api_key_changed = await get_ai_response(messages, model)
        processing_time = time.time() - start_time
        
        # Сохраняем в историю чата, если указан chat_id
        if chat_id:
            if chat_id not in chat_histories:
                chat_histories[chat_id] = []
            
            chat_histories[chat_id].append({"role": "user", "content": f"DeepSearch: {query}"})
            chat_histories[chat_id].append({"role": "assistant", "content": response_text})

        return {
            "response": response_text,
            "chat_id": chat_id,
            "status": "success",
            "processing_time": processing_time,
            "api_key_changed": api_key_changed
        }

    except HTTPException as http_exc:
        raise
    except Exception as e:
        logger.error(f"Server error in deep_search: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

@app.post("/veronica/generate-image", response_model=ChatResponse)
async def generate_image(request: ImageGenerationRequest):
    try:
        prompt = request.prompt
        chat_id = request.chat_id
        model = request.model or IMAGE_GEN_MODEL

        if not prompt:
            raise HTTPException(status_code=400, detail="Prompt is required")

        # Запрос к Hugging Face API для генерации изображения
        headers = {
            "Authorization": f"Bearer {HUGGING_FACE_API_KEY}"
        }

        payload = {
            "inputs": prompt,
            "parameters": {
                "negative_prompt": "blurry, bad quality, distorted",
                "guidance_scale": 7.5,
                "num_inference_steps": 30
            }
        }

        async with aiohttp.ClientSession() as session:
            logger.info(f"Sending image generation request to Hugging Face with model: {model}")
            api_url = f"https://api-inference.huggingface.co/models/{model}"
            
            async with session.post(api_url, headers=headers, json=payload) as response:
                if response.status == 200:
                    # Получаем изображение в виде байтов
                    image_bytes = await response.read()
                    
                    # Сохраняем изображение во временный файл
                    file_id = str(uuid.uuid4())
                    image_filename = f"generated_image_{file_id}.png"
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
                        tmp_file.write(image_bytes)
                        tmp_path = tmp_file.name
                    
                    # Сохраняем информацию о файле
                    uploaded_files[file_id] = {
                        "path": tmp_path,
                        "filename": image_filename,
                        "content_type": "image/png"
                    }
                    
                    file_url = f"/files/{file_id}"
                    response_text = f"![Generated Image]({file_url})\n\nИзображение успешно сгенерировано по запросу: **{prompt}**"
                    
                    # Сохраняем в историю чата, если указан chat_id
                    if chat_id:
                        if chat_id not in chat_histories:
                            chat_histories[chat_id] = []
                        
                        chat_histories[chat_id].append({"role": "user", "content": f"Сгенерировать изображение: {prompt}"})
                        chat_histories[chat_id].append({"role": "assistant", "content": response_text})
                    
                    return {
                        "response": response_text,
                        "chat_id": chat_id,
                        "status": "success"
                    }
                else:
                    error_text = await response.text()
                    logger.error(f"Hugging Face API Error: {error_text}")
                    raise HTTPException(status_code=response.status, detail=f"Hugging Face API Error: {error_text}")

    except HTTPException as http_exc:
        raise
    except Exception as e:
        logger.error(f"Server error in generate_image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
