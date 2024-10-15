import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from gensim.models import Word2Vec
from gensim.similarities import WmdSimilarity
import os
from transformers import pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

class QueryModel(BaseModel):
    query: str

# CSV Path
csv_path = os.path.join(os.path.dirname(__file__), 'qa_base.csv')
model_path = os.path.join(os.path.dirname(__file__), 'word2vec.model')

# Load or train the Word2vec model
try:
    if os.path.exists(model_path):
        model = Word2Vec.load(model_path)
        logger.info("Modelo Word2Vec carregado do disco.")
    else:
        df = pd.read_csv(csv_path)
        logger.info("CSV carregado com sucesso.")
        
        questions = df['Pergunta'].apply(lambda x: x.lower().split()).tolist()
        model = Word2Vec(sentences=questions, vector_size=100, window=5, min_count=1, workers=4)
        model.save(model_path)
        logger.info("Modelo Word2Vec treinado e salvo no disco.")
except Exception as e:
    logger.error(f"Erro ao carregar ou treinar o modelo Word2Vec: {e}")
    raise

# Load the CSV and create pairs-response
try:
    if 'df' not in locals():
        df = pd.read_csv(csv_path)
    questions = df['Pergunta'].apply(lambda x: x.lower().split()).tolist()
    answers = df['Resposta'].tolist()
    qa_pairs = list(zip(questions, answers))
    logger.info("Pré-processamento concluído.")
except KeyError as e:
    logger.error(f"Erro de chave no DataFrame: {e}")
    raise
except Exception as e:
    logger.error(f"Erro no pré-processamento: {e}")
    raise

# Create the similarity index
try:
    index = WmdSimilarity(questions, model.wv, num_best=3)
    logger.info("Índice de similaridade criado com sucesso.")
except Exception as e:
    logger.error(f"Erro ao criar o índice de similaridade: {e}")
    raise

# Function to recover similar answers
def retrieve_info(query):
    query_tokens = query.lower().split()
    try:
        similar_responses = index[query_tokens]
        return [qa_pairs[int(doc_id)][1] for doc_id, sim in similar_responses]
    except Exception as e:
        logger.error(f"Erro ao recuperar informações: {e}")
        raise

# Initialize the Hugging Face text generation pipeline
qa_generator = pipeline('text-generation', model='distilgpt2')

# Function to generate a smart response based on similar answers
def generate_response(query):
    try:
        similar_responses = retrieve_info(query)
        combined_context = " ".join(similar_responses)
        generated_response = qa_generator(f"Baseado no contexto: {combined_context}\nResposta:", max_length=100, num_return_sequences=1)
        return generated_response[0]['generated_text']
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
        raise HTTPException(status_code=500, detail="Erro interno do servidor")

@app.get("/")
async def root():
    return {"message": "API está funcionando"}

@app.get("/health")
async def health_check():
    return {"status": "ok"}
