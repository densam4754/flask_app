from flask import Flask, request, jsonify
import openai
import psycopg2
import os
from dotenv import load_dotenv
from scipy.spatial.distance import cosine

load_dotenv()

app = Flask(__name__)
openai.api_key = os.getenv("OPENAI_API_KEY")
DB_URL = os.getenv("SUPABASE_DB_URL")

def get_embedding(text):
    """Generate OpenAI embedding for query text"""
    response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response['data'][0]['embedding']

def get_all_content_embeddings():
    """Fetch post ID, description, and embedding from DB"""
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()

    cur.execute("""
        SELECT posts.id, posts.description, content_embeddings.embedding
        FROM posts
        JOIN content_embeddings ON posts.id = content_embeddings.post_id
        WHERE posts.description IS NOT NULL
    """)
    results = cur.fetchall()

    cur.close()
    conn.close()
    return results

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    query = data.get("query")

    if not query:
        return jsonify({"error": "Missing search query"}), 400

    try:
        # 1. Embed the query
        query_vector = get_embedding(query)

        # 2. Get all stored content embeddings
        content_rows = get_all_content_embeddings()

        # 3. Calculate similarity
        ranked_results = []
        for post_id, description, embedding in content_rows:
            similarity = 1 - cosine(query_vector, embedding)
            ranked_results.append({
                "post_id": post_id,
                "description": description,
                "score": round(similarity, 4)
            })

        # 4. Sort by similarity
        ranked_results.sort(key=lambda x: x["score"], reverse=True)

        return jsonify(ranked_results[:10])  # Return top 10 matches

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
