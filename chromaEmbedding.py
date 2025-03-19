import chromadb
from sentence_transformers import SentenceTransformer
from torch.nn.functional import embedding

sentences = [
    'A stylish bedroom townhouse close to the river with a private balcony.',
    'A serene 4 bedroom cabin in the mountains with a cozy fireplace.',
    'A sleek studio apartment in a vibrant neighborhood with a shared gym.',
    'A charming 2 bedroom flat near the university with a spacious living area.',
    'A modern 1 bedroom loft in the heart of the city with a rooftop terrace.',
    'A bright and airy studio near the university with large windows.',
    'A cozy 3 bedroom apartment with a garden view near the campus.',
    'A newly renovated flat with natural light and modern furnishings.',
    'A spacious penthouse near the university with a panoramic city view.',
    'A student-friendly apartment within walking distance of campus.',
    'A contemporary 2 bedroom home with a private patio near the university.',
    'A compact and efficient studio with study space near the college.',
    'A luxurious 4 bedroom house with a backyard, close to the university.',
    'A well-lit loft with minimalist decor, just minutes from school.',
    'A quiet retreat near the college with access to public transport.',
    'A chic and modern apartment with floor-to-ceiling windows in a lively area.',
    'A top-floor unit with a scenic balcony near the university library.',
    'A budget-friendly room in a shared house near the university.',
    'A classic-style home with wooden floors and large bedrooms near campus.',
    'A peaceful suburban home within commuting distance of the university.',
    'A student-oriented 2 bedroom apartment with easy access to amenities.',
    'A riverside studio with lots of sunlight and a small balcony.',
    'A newly built dorm-style housing near the university with shared kitchens.',
    'A small but comfortable room near the library and student center.',
    'A modern duplex apartment within a safe student-friendly neighborhood.',
    'A stylish loft with contemporary furniture and a rooftop view.',
    'A well-maintained townhouse near the university with a small garden.',
    'A high-rise condo near the college with a gym and swimming pool.',
    'A bright 1 bedroom apartment in a cultural district near the university.',
    'A charming vintage house with unique architecture and natural lighting.',
    'A trendy 2 bedroom flat near the downtown campus area.',
    'A minimalist apartment with open space and clean aesthetics near campus.',
    'A lakeside villa with plenty of light, ideal for students seeking peace.',
    'A historic home converted into student housing near the university.',
    'A cozy 1 bedroom rental with a spacious desk and bright windows.',
    'A newly built apartment complex catering to university students.',
    'A budget-friendly shared house with social areas near the university.',
    'A single-room apartment with a private entrance and natural light.',
    'A sophisticated condo with modern decor, ideal for student living.',
    'A well-designed 3 bedroom home in a student-oriented neighborhood.',
    'A bright and spacious duplex with an open kitchen and living area.',
    'A contemporary-style residence close to university lecture halls.',
    'A 2 bedroom apartment in a green neighborhood, perfect for studying.',
    'A sleek and modern rental with large windows near the college gym.',
    'A small but efficient studio with a high ceiling near the university.',
    'A university-affiliated housing option with bright rooms and shared spaces.',
    'A stylish shared flat with a spacious common area for students.',
    'A newly built student residence with a rooftop terrace and lounge.',
    'A high-rise student complex with panoramic views and study rooms.',
    'A peaceful studio in a safe neighborhood near the university.',
    'A trendy apartment with modern appliances and excellent lighting.',
    'A lively residence with common areas and bright, cheerful decor.',
    'A fully furnished rental home with large windows and warm lighting.',
    'A small but cozy apartment close to campus, ideal for studying.',
    'A sustainable living space with energy-efficient lighting near the university.',
    'A stylish home with open-concept living and large sliding glass doors.',
    'A smart home apartment near the university with automated lighting.',
    'A charming and rustic student rental with wooden interiors and warmth.',
    'A compact 1 bedroom apartment designed for efficiency and comfort.',
    'A penthouse unit near the university with wraparound windows.',
    'A cozy basement suite with ample lighting near the university campus.',
    'A modern co-living space for students with bright shared areas.',
    'A bright single-bedroom rental with a study-friendly layout.',
    'A comfortable family-style home with student-friendly amenities.',
    'A modern student studio with built-in storage and soft lighting.',
    'A newly developed high-rise offering bright rooms for students.',
    'A quiet top-floor apartment near the university’s science faculty.',
    'A vibrant rental with access to parks, cafes, and bright open spaces.',
    'A fully equipped student apartment with great lighting and decor.',
    'A sleek and minimalistic space near campus, designed for focus.',
    'A smartly arranged small unit with natural light and workspace.',
    'A comfortable university-area rental with plenty of shared amenities.',
    'A bright and colorful student house with fun and inviting decor.',
    'A thoughtfully designed space with maximum natural light exposure.',
    'A modern home with floor-to-ceiling glass windows near campus.',
    'A peaceful apartment with a garden view, designed for students.',
    'A chic micro-apartment that maximizes space and lighting efficiency.',
    'A comfortable and airy rental near the university sports complex.',
    'A high-rise student-friendly apartment with great city views.',
    'A trendy neighborhood loft with vibrant energy and ample sunlight.',
    'A contemporary high-ceiling studio near the campus park.',
    'A spacious and bright rental unit with sleek interior design.',
    'A charming and classic student rental with a cozy atmosphere.',
    'A new student co-living space with vibrant and bright social areas.',
    'A fully furnished apartment with ample workspace and natural light.',
    'A simple and effective layout with a student-focused design.',
    'A friendly and lively student residence with bright common areas.',
    'A contemporary rental with stylish furnishings and warm lighting.',
    'A peaceful student housing option with a quiet and bright interior.',
    'A minimalist yet cozy space optimized for university students.',
    'A studio apartment designed to maximize natural brightness.',
    'A sunny suburban apartment with easy access to the university.',
    'A university-friendly duplex with a bright and spacious interior.',
    'A relaxing student rental with modern lighting and decor touches.',
    'A charming, light-filled cottage near the university gardens.',
    'A cozy attic apartment with skylights near the university center.',
    'A polished and modern apartment with an airy feel near campus.'
]

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings = model.encode(sentences)

client = chromadb.Client()

collection_name = "Apt101"
collection = client.get_or_create_collection(collection_name)

collection.add(
    ids=[str(i) for i in range(len(sentences))],
    embeddings=embeddings.tolist(),
    metadatas=[{"description":desc} for desc in sentences]
)

query = "bright house near a university"
query_embedding = model.encode([query])

print("Embedded query result: ", query_embedding)

results = collection.query(
    query_embeddings=[query_embedding.tolist()[0]],
    n_results=10,
    include=['metadatas','distances']
)

for idx, (metadata, distance) in enumerate(zip(results['metadatas'][0], results['distances'][0])):
    print(f"Result {idx+1}: {metadata['description']} (Distance: {distance})")