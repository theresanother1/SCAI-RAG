import sqlite3
import chromadb
from sentence_transformers import SentenceTransformer
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from helper import get_similarity_model, sanitize

def build_vector_store():
    """
    Builds a persistent vector store from the data in the SQLite database,
    embedding information about students, faculty, and courses.
    """
    # Check if vector store already exists and has data
    try:
        client = chromadb.PersistentClient(path="rag/vector_store")
        collection = client.get_collection("university_data")
        count = collection.count()
        if count > 0:
            print(f"Vector store already exists with {count} documents. Skipping rebuild.")
            return
    except:
        # Collection doesn't exist, create it
        pass
    
    conn = sqlite3.connect('database/university.db')
    cursor = conn.cursor()

    documents = []
    print("Creating student docs")
    # === Build Student Documents ===
    student_query = """
        SELECT s.name, s.email, s.svnr, GROUP_CONCAT(c.name, ', ') AS courses
        FROM students s
        LEFT JOIN enrollments e ON s.id = e.student_id
        LEFT JOIN courses c ON e.course_id = c.id
        GROUP BY s.id
    """
    for name, email, svnr, courses in cursor.execute(student_query).fetchall():
        doc = f"""
        Student Name: {name}
        Email: {email}
        SVNR: {svnr}
        Enrolled Courses: {courses if courses else 'None'}
        """
        documents.append(sanitize(doc.strip()))

    print(documents[-1])

    print("Creating faculty docs")
    # === Build Faculty Documents ===
    faculty_query = """
        SELECT f.name, f.email, f.department, GROUP_CONCAT(c.name, ', ') AS courses
        FROM faculty f
        LEFT JOIN courses c ON f.id = c.faculty_id
        GROUP BY f.id
    """
    for name, email, department, courses in cursor.execute(faculty_query).fetchall():
        doc = f"""
        Faculty Name: {name}
        Email: {email}
        Department: {department}
        Courses Taught: {courses if courses else 'None'}
        """
        documents.append(sanitize(doc.strip()))
    
    print(documents[-1])

    print("Creating course docs")
    # === Build Course Documents ===
    course_query = """
        SELECT 
            c.name as course_name, 
            f.name AS faculty_name,
            f.department AS faculty_department, 
            GROUP_CONCAT(s.name, ', ') AS students
        FROM courses c
        LEFT JOIN faculty f ON c.faculty_id = f.id
        LEFT JOIN enrollments e ON c.id = e.course_id
        LEFT JOIN students s ON e.student_id = s.id
        GROUP BY c.id
    """

    for course_name, faculty_name, faculty_department, students in cursor.execute(course_query).fetchall():
        doc = f"""
        Course Name: {course_name}
        Taught by: {faculty_name if faculty_name else 'TBD'}
        Enrolled Students: {students if students else 'None'}
        Department: {faculty_department if faculty_department else "Unkown"}
        """
        documents.append(doc.strip())


    print(documents[-1])

    conn.close()

    # === Embed and Store in Vector DB ===
    model = get_similarity_model()
    embeddings = model.encode(documents)

    client = chromadb.PersistentClient(path="rag/vector_store")
    collection = client.get_or_create_collection("university_data")

    # Add documents in batches to avoid batch size limits
    batch_size = 5000  # Safe batch size under the limit
    for i in range(0, len(documents), batch_size):
        end_idx = min(i + batch_size, len(documents))
        batch_embeddings = embeddings[i:end_idx]
        batch_documents = documents[i:end_idx]
        batch_ids = [str(j) for j in range(i, end_idx)]
        
        collection.add(
            embeddings=batch_embeddings,
            documents=batch_documents,
            ids=batch_ids
        )

    print("Vector store built successfully.")


if __name__ == "__main__":
    build_vector_store()
