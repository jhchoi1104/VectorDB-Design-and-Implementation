import os
from openai import OpenAI

os.environ["OPENAI_API_KEY"] = ""
client = OpenAI()

def generate_answer(similarity_results):
    formatted_results = "\n".join([f"Result {i +1 }: {result}" for i, result in enumerate(similarity_results)])

    prompt =f"""
    You are an AI assistant. Based on the following similarity search results, provide a helpful response to the user in Korean:
    Similarity Search Results: 
    {formatted_results}
    Answer (Please response in Korea):
    Example
    "안녕하세요.
    첫 번째는 무엇무엇입니다.
    두 번째는 무엇무엇입니다.
    마지막으로, 무엇무엇입니다.
    추가적인 질문이 있으시면 언제든지 말씀해주세요."
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role":"system","content":"You are a helpful assistant."},
            {"role":"user","content": prompt}
        ],
        max_tokens=500,
        temperature=1.0
    )

    answer = response.choices[0].message.content

    return answer

similarity_results = [
    "Result 1: A university-affiliated housing option with bright rooms and shared spaces. (Distance: 0.5304985642433167)",
    "Result 2: A bright 1 bedroom apartment in a cultural district near the university. (Distance: 0.5553813576698303)",
    "Result 3: A smart home apartment near the university with automated lighting. (Distance: 0.5660004615783691)",
    "Result 4: A cozy attic apartment with skylights near the university center. (Distance: 0.5819541811943054)",
    "Result 5: A bright and airy studio near the university with large windows. (Distance: 0.5928825736045837)"
    "Result 6: A sunny suburban apartment with easy access to the university. (Distance: 0.610889196395874)"
    "Result 7: A friendly and lively student residence with bright common areas. (Distance: 0.6226887702941895)"
    "Result 8: A cozy basement suite with ample lighting near the university campus. (Distance: 0.6234497427940369)"
    "Result 9: A contemporary 2 bedroom home with a private patio near the university. (Distance: 0.6235071420669556)"
    "Result 10: A budget-friendly room in a shared house near the university. (Distance: 0.6493912935256958)"
]

answer = generate_answer(similarity_results)
print("Generated Answer:\n", answer)