from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase


model = SentenceTransformer('all-MiniLM-L6-v2')

texts = [
    "그래프 데이터베이스는 노드와 엣지 구조를 사용하여 데이터 간의 관계를 직관적으로 표현하는 데이터베이스입니다.",
    "복잡한 관계형 데이터를 빠르게 탐색할 수 있어 추천 시스템, 소셜 네트워크 분석 등에 자주 활용됩니다.",
    "대표적인 그래프 DB로는 Neo4j, Amazon Neptune, ArangoDB 등이 있으며, 쿼리 언어로는 Cypher가 널리 사용됩니다.",
    "북한산은 서울과 경기 북부에 걸쳐 있는 국립공원으로, 다양한 등산로와 수려한 풍경으로 많은 등산객들이 찾는 명소입니다.",
    "인수봉, 백운대 등 독특한 암봉들이 유명하며, 도심 속에서 자연을 만끽할 수 있는 장소로 인기가 높습니다.",
    "설악산은 강원도에 위치한 대한민국의 대표적인 명산으로, 사계절 내내 아름다운 경치와 웅장한 산세로 많은 관광객이 찾습니다."
]

embeddings = model.encode(texts)

uri = "bolt://localhost:7687"
user = "neo4j"
password = "chlwhdgus12?"

driver = GraphDatabase.driver(uri, auth=(user,password))

# Neo4j에 저장하는 함수 정의
def save_embeddings_to_neo4j(driver, texts, embeddings):
    with driver.session() as session:
        for i, (text, embedding) in enumerate(zip(texts, embeddings)):
            session.run(
                """
                CREATE (n: Mytexts {id: $id, content: $content, embedding: $embedding})
                """,
                id=i,
                content=text,
                embedding=embedding.tolist()
            )

save_embeddings_to_neo4j(driver, texts, embeddings)

driver.close()



