from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
import warnings
import os
from openai import OpenAI

warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.tokenization_utils_base")

CHUNK_SIZE = 256
CHUNK_OVERLAP = 20
EMBEDDING_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
OpenAI_API_KEY = ""
CHROMA_COLLECTION_NAME = "Apt011"

os.environ["OPENAI_API_KEY"] = OpenAI_API_KEY
openai_client = OpenAI()

class TextChunkProcessor:
    def __init__(self, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    def split_text(self, text):
        documents = self.text_splitter.create_documents([text])
        return [doc.page_content for doc in documents]

class TextEmbedder:
    def __init__(self, model_name=EMBEDDING_MODEL_NAME):
        self.model = SentenceTransformer(model_name)

    def generate_embeddings(self, texts):
        return self.model.encode(texts)

class ChromaDBHandler:
    def __init__(self, collection_name):
        self.client = chromadb.PersistentClient()
        self.collection = self.client.get_or_create_collection(collection_name)

    def add_to_collection(self, ids, embeddings, metadatas):
        self.collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas)

    def query_collection(self, query_embedding, n_result=2):
        return self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=n_result,
            include=['metadatas','distances']
        )

def generate_chatgpt_response(query, similarity_results):
    formatted_results="\n".join([f"Result {i+1}: {result}" for i, result in enumerate(similarity_results)])

    prompt = f"""
    당신은 AI 어시스턴트입니다. 다음의 유사도 검색 결과를 기반으로 사용자가 이해할 수 있도록 한국어로 답변하세요:
    유사도 검색 결과:
    {formatted_results}

    답변 (한국어로 작성)
    질문 "{query}에 대한 답변은 아래와 같습니다.
    """

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role":"system","content":"You are a helpful assistant."},
            {"role":"user","content":prompt}
            ],
            max_tokens=500,
            temperature=1.0
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    text = """
    봄날의 산책과 따뜻한 기억들
어느덧 봄이 찾아왔다. 겨우내 차갑고 쓸쓸했던 거리는 이제 조금씩 생기를 되찾고 있다. 거리의 가로수들은 연둣빛 새싹을 틔우고, 공원에는 봄꽃들이 하나둘씩 피어나기 시작했다. 사람들은 두꺼운 외투를 벗어 던지고 가벼운 옷차림으로 거리를 거닌다. 길모퉁이 작은 카페에서는 따뜻한 커피 대신 아이스 아메리카노를 주문하는 사람들이 점점 많아졌다. 거리를 걸을 때마다 부드러운 바람이 피부를 스치고 지나갔다. 바람이 살짝 차갑기는 했지만, 겨울처럼 살을 에는 느낌이 아니라 오히려 상쾌했다. 기분 좋은 햇살 아래에서 천천히 걸으며 주변을 둘러보니, 확실히 모든 것이 겨울과는 달라 보였다.

나는 봄이 오면 습관처럼 공원을 찾는다. 겨울 동안 움츠렸던 몸과 마음을 깨우기 위해 가벼운 산책을 즐긴다. 공원은 계절의 변화를 가장 가까이에서 느낄 수 있는 곳이기 때문이다. 겨울 내내 잿빛이었던 나무들은 어느새 작은 새싹들을 틔우고 있었고, 땅에서는 파릇파릇한 풀들이 돋아나고 있었다. 공원에 도착하자마자 벚꽃나무가 눈길을 사로잡았다. 아직 만개하진 않았지만, 연분홍빛 꽃망울이 하나둘 터지며 아름다운 풍경을 연출하고 있었다. 봄이 본격적으로 시작되면 이 나무들도 만개하여 분홍빛으로 물들겠지. 상상만 해도 설레는 순간이었다.

공원에는 이미 많은 사람들이 나와 있었다. 아이들은 잔디밭에서 뛰어놀고 있었고, 연인들은 서로의 손을 꼭 잡고 산책을 즐겼다. 벤치에는 어르신들이 옹기종기 모여 이야기를 나누고 있었다. 그들의 얼굴에는 온화한 미소가 가득했다. 조깅을 하는 사람들도 곳곳에서 보였다. 가벼운 옷차림을 한 채 빠른 걸음으로 달려가는 그들의 얼굴에는 땀이 송골송골 맺혀 있었다. 그들의 얼굴에는 봄의 생기가 묻어 있었다. 나는 조용히 걸으며 주변을 둘러보았다. 가까운 연못에서는 오리들이 헤엄을 치고 있었고, 물 위에는 나뭇잎이 둥둥 떠다녔다. 그 모습을 보고 있자니 마음이 평온해졌다.

길을 걷다 보니 공원의 한쪽에서 거리 음악가가 연주를 하고 있었다. 그는 기타를 치며 잔잔한 멜로디를 연주하고 있었고, 사람들은 그의 연주를 감상하며 잠시 걸음을 멈추었다. 어떤 아이는 연주에 맞춰 춤을 추었고, 몇몇 사람들은 동전을 상자에 넣으며 응원의 미소를 보냈다. 나 역시 그의 연주를 들으며 잠시 머물렀다. 음악은 언제나 사람들의 마음을 따뜻하게 만들어주는 힘이 있는 것 같다.

나는 천천히 공원을 돌며 여러 풍경들을 감상했다. 한쪽에서는 유모차를 끌고 산책하는 부모들이 있었고, 아이들은 신기한 듯 주변을 두리번거리고 있었다. 한 아이는 작은 꽃을 발견하고는 부모에게 달려가 보여주었다. "엄마, 이 꽃 예뻐!"라는 아이의 말에 엄마는 부드럽게 미소를 지으며 아이를 바라보았다. 봄날의 공원은 이렇게 크고 작은 따뜻한 순간들로 가득했다.

산책을 마치고 공원 근처의 작은 빵집에 들렀다. 이곳은 오래된 가게로, 갓 구운 빵 냄새가 항상 가득하다. 나는 커피 한 잔과 따뜻한 크루아상을 주문했다. 창가 자리에 앉아 지나가는 사람들을 바라보며 커피를 한 모금 마셨다. 봄의 따뜻한 햇살이 유리창 너머로 스며들었고, 나는 이 순간이 참 좋다고 생각했다. 커피의 향이 부드럽게 퍼지며 나를 감싸 안았다. 크루아상을 한 입 베어 물자 바삭하면서도 부드러운 식감이 입안 가득 퍼졌다. 이렇게 작은 여유를 즐길 수 있는 시간들이 삶을 더욱 풍요롭게 만들어 주는 것 같다.

계절이 바뀔 때마다 우리는 자연스럽게 새로운 감정을 느끼게 된다. 겨울이 지나고 봄이 오면, 마음 한구석에서 다시 새로운 시작을 꿈꾸게 된다. 오늘의 산책은 그저 짧은 시간이었지만, 내게는 특별한 기억으로 남을 것 같다. 따뜻한 햇살과 꽃향기, 그리고 길거리의 음악까지. 이런 작은 순간들이 모여 우리의 일상을 더욱 풍요롭게 만들어주는 것이 아닐까?

나는 천천히 커피를 마시며, 또 다른 봄날의 산책을 기대해 본다. 😊🌸
    """

    # 1단계: 텍스트 분할
    processor = TextChunkProcessor()
    sentences = processor.split_text(text)

    # 2단계: 임베딩 생성
    embedder = TextEmbedder()
    embeddings = embedder.generate_embeddings(sentences)

    # 3단계: ChromaDB에 데이터 저장
    chrom_handler = ChromaDBHandler(CHROMA_COLLECTION_NAME)
    chrom_handler.add_to_collection(
        ids=[str(i) for i in range(len(sentences))],
        embeddings=[emb.tolist() for emb in embeddings],
        metadatas=[{"description": sentence} for sentence in sentences]
    )

    # 4단계: 데이터 쿼리
    query = "What is the main theme of this story?"
    query_embedding = embedder.generate_embeddings([query])
    search_results = chrom_handler.query_collection(query_embedding)

    # 5단계: GPT 응답 생성
    metadata_results = search_results['metadatas'][0]
    formatted_results = [result['description'] for result in metadata_results]
    answer = generate_chatgpt_response(query, formatted_results)

    print("\n생성된 응답:\n", answer)