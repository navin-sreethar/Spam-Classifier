from transformers import BertTokenizer, BertForSequenceClassification
import torch
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpamDetectionService:
    def __init__(self):
        self.model_name = "fzn0x/bert-spam-classification-model"
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.model = BertForSequenceClassification.from_pretrained(self.model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()



    @torch.no_grad()
    def predict(self, text: str) -> Tuple[str, float]:
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty")
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.device)
        outputs = self.model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        score = probabilities[0][prediction].item()
        label = 'SPAM' if prediction == 1 else 'HAM'
        return label, score