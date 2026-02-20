import joblib
from preprocessing import clean_text

model = joblib.load('models/model.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')

def predict_category(text: str) -> str:
  """
  Função para prever a categoria de um título de notícia.
  """
  cleaned_text = clean_text(text)
  text_vec = vectorizer.transform([cleaned_text])
  prediction = model.predict(text_vec)
  return prediction[0]