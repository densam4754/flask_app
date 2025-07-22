import os
import google.generativeai as genai
import psycopg2
from dotenv import load_dotenv

# Load credentials
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# DB connection
conn = psycopg2.connect(
    dbname=os.getenv("SUPABASE_DB_NAME"),
    user=os.getenv("SUPABASE_DB_USER"),
    password=os.getenv("SUPABASE_DB_PASSWORD"),
    host=os.getenv("SUPABASE_DB_HOST"),
    port="5432"
)
cursor = conn.cursor()

# Step 1: Fetch all posts
cursor.execute("SELECT id, description FROM posts")
posts = cursor.fetchall()

model = genai.GenerativeModel("embedding-001")  # Gemini supports embedding

for post_id, description in posts:
    try:
        response = model.embed_content(content=description, task_type="retrieval_document")
        embedding = response["embedding"]

        # Step 2: Insert into content_embeddings
        cursor.execute("""
            INSERT INTO content_embeddings (post_id, embedding)
            VALUES (%s, %s)
            ON CONFLICT (post_id) DO UPDATE SET embedding = EXCLUDED.embedding
        """, (post_id, embedding))

        print(f"✅ Embedded Post {post_id}")
    except Exception as e:
        print(f"❌ Error embedding post {post_id}: {e}")

conn.commit()
cursor.close()
conn.close()
