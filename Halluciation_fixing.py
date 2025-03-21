from openai import OpenAI
import os
from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase
from numpy import dot
from numpy.linalg import norm


os.environ["OPENAI_API_KEY"] = ""
client = OpenAI()

model = SentenceTransformer('all-MiniLM-L6-v2')

uri = "bolt://localhost:7687"
user = "neo4j"
password = "chlwhdgus12?"

query = "북한산 설명해줘"
# query = "Explain 북한산"
query_embedding = model.encode([query])[0]

def cosine_similarity(vec1, vec2):
    return dot(vec1, vec2) / (norm(vec1)*norm(vec2))

# 쿼리와 모든 원소들의 유사도 계산
# 가장 유사도가 높은 3개 원소 추출
def find_similar_texts(driver, query_embedding, top_n=3):
    with driver.session() as session:
        # result = session.run("match(n:Mytexts) return n.content as content, n.embedding as embedding")
        result = session.run("match(m: Mountain{name:'북한산'})-[: related_to]->(n:Mytexts) return n.content as content, n.embedding as embedding")

        similarities = []

        for record in result:
            content = record["content"]
            embedding = record["embedding"]
            similarity = cosine_similarity(query_embedding, embedding)
            similarities.append((content, similarity))

        similarities = sorted(similarities, key=lambda x:x[1], reverse=True)
        return similarities

def generate_response(similarity_results):
    formatted_results = "\n".join([f"Result {i+1}: {result}" for i, result in enumerate(similarity_results)])

    print(formatted_results)

    prompt = f"""
    You are an AI assistant. Based on the following similarity search results, provide a helpful response to the user in Korean: Similarity Search Results:
    {formatted_results}
    Example :
    안녕하세요. 보내주신 유사도 검색 결과를 기반으로 분석해드리겠습니다.
    xxx는 xxx이고, xxx이며, xxx입니다.
    이상 답변드립니다!
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role":"system", "content": "You are a helpful assistant."},
            {"role":"user", "content":prompt }
        ],
        max_tokens=500,
        temperature=1.0
    )

    answer = response.choices[0].message.content
    return answer

driver = GraphDatabase.driver(uri, auth=(user, password))
similar_texts = find_similar_texts(driver, query_embedding)

response = generate_response(similar_texts)
print(f"Generated Response: {response}")

driver.close()

