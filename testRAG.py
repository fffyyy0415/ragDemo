from docx import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from elasticsearch import Elasticsearch
import requests
import numpy as np
import tiktoken
from openai import OpenAI
encoder = tiktoken.get_encoding("cl100k_base")

def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    try:
        return len(encoder.encode(string))
    except Exception:
        return 0


def truncate(string: str, max_len: int) -> str:
    """Returns truncated text if the length of text exceed max_len."""
    return encoder.decode(encoder.encode(string)[:max_len])
# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://10.12.7.56:11434/api/embed"

# 1. 读取Word文档
def read_docx(file_path):
    """读取Word文档并返回纯文本内容"""
    doc = Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])


# 2. 文本分块
def split_text(text):
    """使用递归字符分割器进行文本分块"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # 每个块的最大字符数
        chunk_overlap=50,  # 块之间的重叠字符数
        length_function=len
    )
    return text_splitter.split_text(text)


# 3. 生成嵌入向量
def generate_embeddings(texts):
    """使用sentence-transformers生成文本嵌入"""

    arr = []
    tks_num = 0
    for txt in texts:
        response = requests.post(
            openai_api_base,
            json={
                "model": "bge-m3",
                "input": txt
            }
        )
        arr.append(response.json()["embeddings"])

    return np.array(arr)






# 5. 创建ES索引
def create_index(es_client, index_name="doc_chunks"):
    """创建包含向量字段的ES索引"""
    if  es_client.indices.exists(index=index_name):
        es_client.indices.create(
            index=index_name,
            body={
    "mappings": {
        "properties": {
            "content": {"type": "text"},
            "embedding": {"type": "dense_vector", "dims": 1024, "similarity": "cosine"}
        }
    }
}

        )


# 6. 存储到Elasticsearch
def store_in_es(es_client, index_name, chunks, embeddings):
    """将分块文本和嵌入存储到ES"""
    for i, (text, embedding) in enumerate(zip(chunks, embeddings)):
        try:
            doc = {'id': i, 'content': text, 'embedding': embedding[0].tolist()}
            res = es_client.index(index='doc_chunks', id= i, body=doc)

            if res['result'] != 'created':
                print(f"Failed to index document {i}")
        except Exception as e:
            print(f"Error indexing document {i}: {e}")


def search_es(es_client, index_name, query_embedding, top_k=10):
    """使用余弦相似度进行向量搜索"""
    response = es_client.search(
        index=index_name,
        body={
            "query": {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                        "params": {"query_vector": query_embedding}
                    }
                }
            },
            "size": top_k
        }
    )



    return [hit["_source"]["content"] for hit in response["hits"]["hits"]]


# 8. 重排序
def rerank_results(query, texts):
    """使用交叉编码器进行结果重排序"""
    # 加载rerank模型
    if len(texts) == 0:
        return np.array([]), 0
    pairs = [(query, truncate(t, 4096)) for t in texts]
    token_count = 0
    for _, t in pairs:
        token_count += num_tokens_from_string(t)
    model_name = "bce-reranker-base_v1"
    base_url = "http://10.12.7.56:9998/v1/rerank"
    headers = {
        "Content-Type": "application/json",
        "accept": "application/json",
        "Authorization": f"Bearer"
    }

    data = {
        "model": model_name,
        "query": query,
        "return_documents": "true",
        "return_len": "true",
        "documents": texts
    }
    res = requests.post(base_url, headers=headers, json=data).json()
    rank = np.zeros(len(texts), dtype=float)
    for d in res["results"]:
        rank[d["index"]] = d["relevance_score"]

    return rank, token_count
def chat_llm(query, recalled_chunks):
    client = OpenAI(
        api_key="EMPTY",
        base_url="http://10.12.39.53:8000/v1",
    )



    prompt_prefix = "你是一个智能助手，请总结知识库的内容来回答问题，请列举知识库中的数据详细回答。当所有知识库内容都与问题无关时，你的回答必须包括“知识库中未找到您要的答案！”这句话。回答需要考虑聊天历史。以下是知识库："
    prompt_prefix2="   以上是知识库。"
    prompt = "{}{}{}{}".format(prompt_prefix, recalled_chunks,prompt_prefix2,query)
    conversation = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]


    response=client.chat.completions.create(
        model="Qwen2.5-7B-Instruct-AWQ",
        messages=conversation,
        temperature=0.6,
        top_p=0.8,
        # max_tokens=4096,
        extra_body={
            "repetition_penalty":1.05,
        }
    )
    return response.choices[0].message.content



# 主流程
def main(file_path):
    # # # 1. 读取文档
    # text = read_docx(file_path)
    # #
    # # # 2. 文本分块
    # chunks = split_text(text)
    # #
    # # # 3. 生成嵌入
    # embeddings = generate_embeddings(chunks)
    # #
    # # 4. 连接ES
    es = Elasticsearch("http://10.12.7.56:9200",
                       basic_auth=("elastic", "infini_rag_flow"),
                       verify_certs=False,  # 不验证服务器证书（不推荐）
                       ca_certs=None,
    timeout=600)
    # # 5. 创建索
    # create_index(es)
    # #
    # # # 6. 存储数据
    # store_in_es(es, "doc_chunks", chunks, embeddings)

    # 示例查询
    query = "S21前摄像头多少像素？"

    # 生成查询嵌入
    query_embedding = generate_embeddings([query]).tolist()

    # 召回结果
    recalled_results = search_es(es, "doc_chunks", query_embedding[0][0])

    # 重排序
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
    print(final_results)

if __name__ == "__main__":
    main("1、移动执法实施交付方案-V.docx")
