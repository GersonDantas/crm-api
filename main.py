from flask import Flask, request, jsonify
import pandas as pd
from gensim.models import Word2Vec
from gensim.similarities import WmdSimilarity

app = Flask(__name__)

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

@app.route('/ask', methods=['POST'])
def ask_question():
    query = request.json.get('query')
    if not query:
        return jsonify({'error': 'No query provided'}), 400
    
    query_tokens = query.split()
    similar_responses = index[query_tokens]  # Pegando os 3 documentos mais similares
    
    threshold = 0.7  # Defina um limiar de similaridade adequado para sua aplicação
    responses = [qa_pairs[int(doc_id)][1] for doc_id, sim in similar_responses if sim > threshold]
    
    if not responses:
        # Armazene a pergunta e redirecione se necessário
        # Aqui você pode implementar a lógica para armazenar a pergunta no Wix
        # Vamos apenas retornar uma mensagem de exemplo por enquanto
        return jsonify({'response': 'Pergunta não encontrada. Será redirecionada para o armazenamento.'}), 200
    
    response = " ".join(responses)  # Pode melhorar a concatenação ou lógica de resposta aqui
    return jsonify({'response': response}), 200

if __name__ == '__main__':
    app.run(debug=True)
