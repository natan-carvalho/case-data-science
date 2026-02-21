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

@app.get("/")
def health_check():
  return {"mensagem": "API rodando!"}

@app.post("/predicao")
def predict(request: NewsRequest):
  texto = clean_text(request.title)
  texto_vectorizado = vectorizer.transform([texto])
  predicao = model.predict(texto_vectorizado)
  return {
    "title": request.title,
    "categoria_predita": predicao[0]
  }