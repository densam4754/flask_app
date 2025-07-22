from flask import Flask, request, jsonify
from your_gemini_embedder import get_embedding_from_gemini  # your embedding function
import psycopg2

app = Flask(__name__)

@app.route("/recommend", methods=["POST"])
def recommend():
    query = request.json.get("query")

    if not query:
        return jsonify({"error": "Missing query"}), 400

    embedding = get_embedding_from_gemini(query)

    # Connect to Supabase and call RPC
    conn = psycopg2.connect(...)
    cur = conn.cursor()
    cur.execute("select * from match_posts_by_embedding(%s, %s, %s)", (embedding, 0.3, 5))
    results = cur.fetchall()
    cur.close()
    conn.close()

    return jsonify([
        {"post_id": r[0], "description": r[1], "similarity": r[2]} for r in results
    ])

if __name__ == "__main__":
    app.run(port=5000)
