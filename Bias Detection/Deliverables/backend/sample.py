from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
model_path = "/Users/Adith/Downloads/final_model.pth"
# Load your trained weights
model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
model.eval()

text = "The new government initiative has significantly reduced unemployment in the past quarter, with thousands of people entering the workforce. Citizens are praising the administration’s efforts as a major success"
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=1).item()

print("Prediction:", prediction)
