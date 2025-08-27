from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

class SpamDetectionService:
    def __init__(self):
        self.model_name = "mrm8488/bert-tiny-finetuned-sms-spam-detection"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)

    def predict(self, text: str) -> tuple[str, float]:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        outputs = self.model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
        prediction = torch.argmax(probabilities, dim=1)
        
        label = "SPAM" if prediction.item() == 1 else "HAM"
        score = probabilities[0][prediction.item()].item()
        
        return label, score