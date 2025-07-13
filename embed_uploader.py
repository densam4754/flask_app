import os
import openai
import psycopg2
from dotenv import load_dotenv

# Load .env file for keys and DB
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
db_url = os.getenv("SUPABASE_DB_URL")  # e.g. postgresql://user:pass@host:port/dbname

def get_embedding(text):
    """Generate OpenAI embedding from given text."""
    result = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=text
    )
    return result['data'][0]['embedding']

def embed_posts():
    conn = psycopg2.connect(db_url)
    cur = conn.cursor()

    # Step 1: Fetch all posts with description
    cur.execute("SELECT id, description FROM posts WHERE description IS NOT NULL")
    posts = cur.fetchall()

    for post_id, description in posts:
        try:
            print(f"Embedding post {post_id}: {description[:60]}...")
            embedding = get_embedding(description)

            # Step 2: Insert or update embedding
            cur.execute("""
                INSERT INTO content_embeddings (post_id, embedding)
                VALUES (%s, %s)
                ON CONFLICT (post_id) DO UPDATE
                SET embedding = EXCLUDED.embedding
            """, (post_id, embedding))
        except Exception as e:
            print(f"⚠️ Failed to embed post {post_id}: {e}")

    conn.commit()
    cur.close()
    conn.close()
    print("✅ All post embeddings saved successfully.")

if __name__ == "__main__":
    embed_posts()
