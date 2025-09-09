from sentence_transformers import util
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AutoTokenizer, AutoModelForSequenceClassification, BertTokenizer
import spacy
import numpy as np
import torch
import json
import os
from datetime import datetime, timezone
import asyncio
import logging

logging.basicConfig(level=logging.INFO)
# Replace with your actual model import
# from your_model_file import YourOmissionBiasModelClass

# Constants
OMISSION_BIAS_MODEL_PATH = "/Users/amalkurian/Desktop/Dissertation/Bias Detection/models/DistilBERT/final_model_new(2).pth"
ANALYTICS_FILE = "analytics.json"

# FastAPI app setup
app = FastAPI(title="Omission Bias Detection API")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # <-- Be strict in production!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from newspaper import Article, Config

@app.get("/extract/")
async def extract_article(url: str):
    # print(f"Received URL: {url}")
    try:
        config = Config()
        config.browser_user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'

        article = Article(url, config=config)
        article.download()
        article.parse()
        return {"article": article.text}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Extraction failed: {str(e)}")



# Input schema
class ArticleRequest(BaseModel):
    article_text_1: str
    article_text_2: str


# Lazy loading model and tokenizer
omission_model = None
tokenizer = None

def get_model():
    global omission_model, tokenizer
    if omission_model is None or tokenizer is None:
        # model_name = "./deberta-v2-base"

        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        omission_model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert/distilbert-base-uncased", num_labels=2
        )
        state_dict = torch.load(OMISSION_BIAS_MODEL_PATH, map_location=torch.device("cpu"))
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("omissionmodel."):
                new_key = k[len("omissionmodel."):]  # Remove prefix
            else:
                new_key = k
            new_state_dict[new_key] = v
        omission_model.load_state_dict(new_state_dict)
        omission_model.eval()
    return omission_model, tokenizer


def load_analytics():
    if not os.path.exists(ANALYTICS_FILE):
        logging.info(f"Analytics file not found. Creating new one at {ANALYTICS_FILE}")
        return {"total_predictions": 0, "bias_count": 0, "no_bias_count": 0, "logs": []}
    try:
        with open(ANALYTICS_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logging.warning(f"Analytics file corrupted or unreadable: {e}. Resetting.")
        return {"total_predictions": 0, "bias_count": 0, "no_bias_count": 0, "logs": []}


def save_analytics(data):
    temp_file = ANALYTICS_FILE + ".tmp"
    with open(temp_file, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(temp_file, ANALYTICS_FILE)


# # Main prediction route
nlp = spacy.load("en_core_web_sm")


embedding_model = None
def get_embedding_model():
    global embedding_model
    if embedding_model is None:
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    return embedding_model



#    embeddings = embed_model.encode([request.article_text_1, request.article_text_2], convert_to_tensor=True)
#    # emb1, emb2 = embeddings[0], embeddings[1]
     # def safe_cosine_sim(a, b):
    #     print("Calculating cosine similarity")
    #     norm_a = np.linalg.norm(a)
    #     norm_b = np.linalg.norm(b)
    #     print(f"norm_a: {norm_a}, norm_b: {norm_b}")
    #     if norm_a == 0 or norm_b == 0:
    #         print("One of the vectors is zero vector")
    #         return 0.0
    #     result = np.dot(a, b) / (norm_a * norm_b)
    #     print(f"Cosine similarity: {result}")
    #     return result

    # cosine_sim = safe_cosine_sim(emb1, emb2)

# ---- Entity similarity helpers ----

def extract_entities(text):
    """Extract named entities from text using spaCy."""
    doc = nlp(text)
    return [ent.text.lower().strip() for ent in doc.ents]

def jaccard_entity_similarity(ents1, ents2):
    """Compute entity similarity using Jaccard index."""
    set1, set2 = set(ents1), set(ents2)
    if not set1 and not set2:
        return 1.0  # both empty = perfect match
    if not set1 or not set2:
        return 0.0  # one empty = no overlap
    return len(set1 & set2) / len(set1 | set2)

def embedding_entity_similarity(ents1, ents2, model):
    """Compute entity similarity by embedding and comparing."""
    if not ents1 or not ents2:
        return 0.0
    emb1 = model.encode(ents1, convert_to_tensor=True)
    emb2 = model.encode(ents2, convert_to_tensor=True)
    sim_matrix = util.cos_sim(emb1, emb2)  # [len(ents1), len(ents2)]
    # for each entity in ents1, take max similarity in ents2
    max_scores = sim_matrix.max(dim=1).values
    return max_scores.mean().item()



@app.post("/predict-omission/")
async def predict_omission(request: ArticleRequest):
    model, tokenizer = get_model()
    embed_model = get_embedding_model()

    def analyze_article(text):
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        )
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            prediction = torch.argmax(logits, dim=1).item()
        return prediction

    pred1 = analyze_article(request.article_text_1)
    logging.info(f"Prediction for article 1: {pred1}")
    pred2 = analyze_article(request.article_text_2)
    logging.info(f"Prediction for article 2: {pred2}")
    prediction_flag = 0 if (pred1 != pred2 or pred1 == 0) else 1  # 1 means no omission bias

    # Use SentenceTransformer encode with convert_to_tensor=True for util.cos_sim
    embeddings = embed_model.encode(
        [request.article_text_1, request.article_text_2], convert_to_tensor=True
    )

   # emb1, emb2 = embeddings[0], embeddings[1]
    def safe_cosine_sim(a, b):
        print("Calculating cosine similarity")
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        print(f"norm_a: {norm_a}, norm_b: {norm_b}")
        if norm_a == 0 or norm_b == 0:
            print("One of the vectors is zero vector")
            return 0.0
        result = np.dot(a, b) / (norm_a * norm_b)

        # cosine_sim = util.cos_sim(a, b)
        # If you just want the scalar value (not a tensor)
        # result = cosine_sim.item()
        print(f"Cosine similarity: {result}")
        return result

    cosine_sim = safe_cosine_sim(embeddings[0], embeddings[1]).item()


    logging.info(f"Received article 1: {request.article_text_1[:200]}")  # log first 200 chars
    logging.info(f"Received article 2: {request.article_text_2[:200]}")

    entities1 = extract_entities(request.article_text_1)
    entities2 = extract_entities(request.article_text_2)
    entities2 = list(set(entities1) & (set(entities2)))

    logging.info(f"Cosine similarity computed: {cosine_sim}")

    analytics = load_analytics()
    analytics["total_predictions"] += 2
    if prediction_flag == 1:
        analytics["bias_count"] += 1
    else:
        analytics["no_bias_count"] += 1

    analytics["logs"].append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "prediction_agreement": bool(prediction_flag),
        "cosine_similarity": cosine_sim,
    })
    save_analytics(analytics)

    return{
        "bias_result": "Bias Detected" if prediction_flag == 0 else "No Bias",
        "prediction_agreement": bool(prediction_flag),
        "cosine_similarity": cosine_sim,
        "entities_article_1": entities1,
        "entities_article_2": entities2
    }
   


# Analytics route
@app.get("/analytics/")
async def get_analytics():
    return load_analytics()

# Reset route
@app.post("/reset/")
async def reset_analytics():
    default_data = {
        "total_predictions": 0,
        "bias_count": 0,
        "no_bias_count": 0,
        "logs": []
    }
    save_analytics(default_data)
    return {"message": "Analytics reset successfully."}

# Basic health check
@app.get("/")
async def root():
    return {"message": "Omission Bias API is live."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
