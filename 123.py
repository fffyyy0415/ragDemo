from flask import Flask, request, jsonify, make_response, send_from_directory
from testRAG import generate_embeddings, search_es, rerank_results, chat_llm, Elasticsearch
app = Flask(__name__)

@app.route('/')
def index():
    return send_from_directory('static', 'upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'fileInput' not in request.files:
        return jsonify({'message': 'No file part'}), 400
    file = request.files['fileInput']
    if file.filename == '':
        return jsonify({'message': 'No selected file'}), 400
    if file:
        # 这里可以添加保存文件的逻辑
        response = make_response(jsonify({'message': 'File successfully uploaded'}), 200)
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response

@app.route('/sendData', methods=['POST'])
def send_data():
    es = Elasticsearch("http://10.12.7.56:9200",
                       basic_auth=("elastic", "infini_rag_flow"),
                       verify_certs=False,  # 不验证服务器证书（不推荐）
                       ca_certs=None,
                       timeout=600)
    query = request.json.get('data')
    query_embedding = generate_embeddings([query]).tolist()
    recalled_results = search_es(es, "doc_chunks", query_embedding[0][0])
    final_results = rerank_results(query, recalled_results)
    recalled_chunks = []
    recalled_socre = []
    for i in range(len(final_results[0])):
        if final_results[0][i] > 0.5:
            recalled_chunks.append(recalled_results[i])
            recalled_socre.append(final_results[0][i])
    print(recalled_chunks)
    print(recalled_socre)
    # 聊天
    final_results = chat_llm(query, recalled_chunks)
    response = make_response(final_results)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

if __name__ == '__main__':
    app.run(debug=True)