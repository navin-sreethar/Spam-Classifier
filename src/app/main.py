from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from .service import SpamDetectionService  # Changed to relative import
import uvicorn

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

app = FastAPI()
templates = Jinja2Templates(directory="src/app/templates")
spam_detector = SpamDetectionService()

@app.get("/", response_class=HTMLResponse)
async def get_form(request: Request):
    return templates.TemplateResponse(
        "form.html",
        {"request": request, "result": None}
    )

@app.post("/", response_class=HTMLResponse)
async def predict_spam(request: Request, email: str = Form(...)):
    label, score = spam_detector.predict(email)
    return templates.TemplateResponse(
        "form.html",
        {
            "request": request,
            "result": {"text": email, "label": label, "score": f"{score:.2%}"}
        }
    )