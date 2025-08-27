# Email Spam Detection App

A modern web application that detects spam emails using AI. Built with FastAPI and BERT, this project demonstrates practical machine learning integration in a web service.

## Features

- Real-time spam detection using BERT model
- Fast and accurate predictions with confidence scores
- Clean and intuitive web interface
- Production-ready architecture

## Tech Stack

- **Backend**: FastAPI + Python 3.11
- **ML Model**: BERT (via HuggingFace Transformers)
- **Frontend**: HTML + Jinja2 Templates
- **Core Libraries**: PyTorch, Transformers

## Quick Start

1. Set up your environment:
   ```bash
   # Create and activate virtual environment
   python3 -m venv venv
   source venv/bin/activate

   # Install dependencies
   pip install -r requirements.txt
   ```

2. Run the application:
   ```bash
   uvicorn src.app.main:app --reload
   ```

3. Open `http://localhost:8000` in your browser

## Project Structure

```
spam-detection-api/
├── src/
│   └── app/
│       ├── main.py          # FastAPI routes and app logic
│       ├── service.py       # Spam detection service
│       └── templates/
│           └── form.html    # Web interface
└── requirements.txt
```

## How It Works

1. User submits email content through the web form
2. BERT model analyzes the text
3. App returns prediction (SPAM/HAM) with confidence score

## Potential Enhancements

- Add API documentation
- Include model performance metrics
- Add Docker support
- Implement batch processing