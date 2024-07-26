import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from gensim.models import Word2Vec
from gensim.similarities import WmdSimilarity
import os

# Configuring the logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

class QueryModel(BaseModel):
    query: str

# CSV Path
csv_path = os.path.join(os.path.dirname(__file__), 'qa_base.csv')

# Load the CSV using pandas
try:
    df = pd.read_csv(csv_path)
    logger.info("CSV carregado com sucesso.")
except Exception as e:
    logger.error(f"Erro ao carregar o CSV: {e}")
    raise

# Pre-processing of questions and answers
try:
    questions = df['Pergunta'].apply(lambda x: x.split()).tolist()
    answers = df['Resposta'].tolist()
    qa_pairs = list(zip(questions, answers))
    logger.info("Pré-processamento concluído.")
except KeyError as e:
    logger.error(f"Erro de chave no DataFrame: {e}")
    raise
except Exception as e:
    logger.error(f"Erro no pré-processamento: {e}")
    raise

# Training the Word2vec model
try:
    model = Word2Vec(sentences=questions, vector_size=100, window=5, min_count=1, workers=4)
    logger.info("Modelo Word2Vec treinado com sucesso.")
except Exception as e:
    logger.error(f"Erro ao treinar o modelo Word2Vec: {e}")
    raise

# Creating the similarity index
try:
    index = WmdSimilarity(questions, model.wv, num_best=3)
    logger.info("Índice de similaridade criado com sucesso.")
except Exception as e:
    logger.error(f"Erro ao criar o índice de similaridade: {e}")
    raise

def retrieve_info(query):
    query_tokens = query.split()
    try:
        similar_response = index[query_tokens]  # Taking the 3 most similar documents
        return [qa_pairs[int(doc_id)][1] for doc_id, sim in similar_response]
    except Exception as e:
        logger.error(f"Erro ao recuperar informações: {e}")
        raise

def generate_response(query):
    try:
        similar_responses = retrieve_info(query)
        response = " ".join(similar_responses)  # Here we can improve the response generated
        return response
    except Exception as e:
        logger.error(f"Erro ao gerar resposta: {e}")
        raise

@app.post("/ask")
async def ask_question(query: QueryModel):
    try:
        response = generate_response(query.query)
        return {"response": response}
    except Exception as e:
        logger.error(f"Erro na rota /ask: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.get("/")
async def root():
    return {"message": "API está funcionando"}

@app.get("/health")
async def health_check():
    return {"status": "ok"}
