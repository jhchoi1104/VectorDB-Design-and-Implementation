from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase

texts = [
    "Neo4j is a graph database.",
    "Graph database are great for connected data.",
    "Machine learning can create embeddings for test."
]

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(texts)

uri = "bolt://localhost:7687"
user = "neo4j"
password = "chlwhdgus12?"
driver = GraphDatabase.driver(uri, auth=(user,password))

# 서버를 띄어놓아야 하는 과정이 있을 것 같다.
def save_embeddings_to_neo4j(driver, texts, embeddings):
    with driver.session() as session:
        for i, (text, embedding) in enumerate(zip(texts,embeddings)):
            session.run(
                """
                CREATE(n:Text {id:$id, content: $content, embedding:$embedding})
                """,
                id=i,
                content=text,
                embedding=embedding.tolist()
            )

save_embeddings_to_neo4j(driver, texts, embeddings)

driver.close()

