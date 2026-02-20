from pydantic import BaseModel
from fastapi import FastAPI
import joblib
import sys

sys.path.append('src')

from preprocessing import clean_text

app = FastAPI(title="API de Classificação de Notícias")

model = joblib.load('models/model.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')

class NewsRequest(BaseModel):
  title: str

@app.post("/")
def health_check():
  return {"message": "API is running"}