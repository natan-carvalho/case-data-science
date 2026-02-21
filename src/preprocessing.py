import re
import string

def clean_text(text: str) -> str:
  """
  Função para limpeza básica de texto,
  Remove pontuação, números e transforma para minúsculas.
  """
  text = text.lower()  # Converte para minúsculas
  text = re.sub(r'\d+', '', text)  # Remove números
  text = text.translate(str.maketrans('', '', string.punctuation))  # Remove pontuação
  text = text.strip()  # Remove espaços em branco no início e no final
  return text
