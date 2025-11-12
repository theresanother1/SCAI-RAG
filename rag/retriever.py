import chromadb
from sentence_transformers import SentenceTransformer
import sqlite3
from pathlib import Path
from helper import get_similarity_model, sanitize

def search(query: str, top_k: int = 10):
    """
    Searches the vector store for the most relevant documents to a given query.
    """
    # Handle special case for listing all students
    if "give me the names of the students" in query.lower():
        # Use absolute path for database
        db_path = Path(__file__).parent.parent / "database" / "university.db"
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        students = cursor.execute("SELECT name FROM students").fetchall()
        conn.close()
        return [{"name": student[0]} for student in students]

    # get SentenceTransformer model
    model = get_similarity_model()

    # Create the query embedding - model from helper instantiated only once 
    query_embedding = model.encode([query])

    # Initialize ChromaDB client and get the collection with absolute path
    vector_store_path = Path(__file__).parent / "vector_store"
    client = chromadb.PersistentClient(path=str(vector_store_path))
    collection = client.get_collection("university_data")

    # Perform the search
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )
    sanitized_context = [sanitize(doc) for doc in results['documents'][0]]
    return sanitized_context


if __name__ == '__main__':
    # Example usage
    test_query = "What courses are available?"
    results = search(test_query)
    print(f"Results for '{test_query}':")
    for result in results:
        print(result)

    test_query_students = "Give me the names of the students"
    results_students = search(test_query_students)
    print(f"\nResults for '{test_query_students}':")
    for student in results_students:
        print(student['name'])
