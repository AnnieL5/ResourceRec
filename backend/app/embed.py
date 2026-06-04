from sentence_transformers import SentenceTransformer
import json
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")

student_text = """
Goal: Learn Python for AI
Level: Beginner
Interests: machine learning, programming
"""

student_embedding = model.encode(student_text)

def load_resources(student_embed):
    with open('../data/resource.json', 'r') as file:
        resource_data = json.load(file)

    scores = []
    
    for resource in resource_data:
        resource_embed = model.encode(resource)
        similarity = cosine_similarity([student_embed], [resource_embed])
        scores.append((resource, similarity))

    scores.sort(reverse=True)
    return scores[:5]

print(load_resources(student_embed=student_embedding))
