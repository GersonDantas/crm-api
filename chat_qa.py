import streamlit as st
import pandas as pd
from gensim.models import Word2Vec
from gensim.similarities import WmdSimilarity

# Carregar o CSV usando pandas
df = pd.read_csv("qa_base.csv")

# Pré-processamento das perguntas e respostas
questions = df['Pergunta'].apply(lambda x: x.split()).tolist()
answers = df['Resposta'].tolist()
qa_pairs = list(zip(questions, answers))

# Treinando o modelo Word2Vec
model = Word2Vec(sentences=questions, vector_size=100, window=5, min_count=1, workers=4)

# Criando o índice de similaridade
index = WmdSimilarity(questions, model.wv, num_best=3)

def retrieve_info(query):
    query_tokens = query.split()
    similar_response = index[query_tokens]  # Pegando os 3 documentos mais similares
    return [qa_pairs[int(doc_id)][1] for doc_id, sim in similar_response]

def generate_response(query):
    similar_responses = retrieve_info(query)
    # Aqui podemos simplesmente concatenar as respostas ou escolher uma para exibir
    response = " ".join(similar_responses)
    return response

# Exemplo de uso no Streamlit
st.title("Sistema de Perguntas e Respostas")

query = st.text_input("Faça sua pergunta:")
if query:
    response = generate_response(query)
    st.write(response)
