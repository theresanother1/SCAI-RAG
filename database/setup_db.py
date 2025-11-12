import sqlite3
from faker import Faker
import random

def generate_svnr():
    """
    Generates a valid Austrian social security number (SVNR).
    """
    weights = [3, 7, 9, 0, 5, 8, 4, 2, 1, 6]
    while True:
        # Generate a 9-digit random number for the main part, ensuring the first digit is not 0.
        main_part = str(random.randint(1, 9)) + "".join([str(random.randint(0, 9)) for _ in range(8)])
        
        # Calculate the checksum from the main part
        partial_sum = (int(main_part[0]) * weights[0] +
                       int(main_part[1]) * weights[1] +
                       int(main_part[2]) * weights[2] +
                       int(main_part[3]) * weights[4] +
                       int(main_part[4]) * weights[5] +
                       int(main_part[5]) * weights[6] +
                       int(main_part[6]) * weights[7] +
                       int(main_part[7]) * weights[8] +
                       int(main_part[8]) * weights[9])
        
        checksum = partial_sum % 11
        
        # The checksum must be a single digit. If it's 10, the number is invalid, so we regenerate.
        if checksum < 10:
            svnr = main_part[:3] + str(checksum) + main_part[3:]
            return svnr

def setup_database():
    """
    Initializes and populates the SQLite database with synthetic data.
    """
    conn = sqlite3.connect('database/university.db')
    cursor = conn.cursor()

    # Create tables
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS students (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        email TEXT NOT NULL UNIQUE,
        svnr TEXT NOT NULL UNIQUE
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS faculty (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        email TEXT NOT NULL UNIQUE,
        department TEXT NOT NULL
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS courses (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        faculty_id INTEGER,
        FOREIGN KEY (faculty_id) REFERENCES faculty (id)
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS enrollments (
        student_id INTEGER,
        course_id INTEGER,
        PRIMARY KEY (student_id, course_id),
        FOREIGN KEY (student_id) REFERENCES students (id),
        FOREIGN KEY (course_id) REFERENCES courses (id)
    )
    ''')

    # Populate with synthetic data
    fake = Faker()
    
    # Add students
    for _ in range(500):
        try:
            cursor.execute("INSERT INTO students (name, email, svnr) VALUES (?, ?, ?)", 
                           (fake.name(), fake.email(), generate_svnr()))
        except sqlite3.IntegrityError:
            pass

    # Add faculty
    for _ in range(100):
        try:
            cursor.execute("INSERT INTO faculty (name, email, department) VALUES (?, ?, ?)", 
                           (fake.name(), fake.email(), fake.job()))
        except sqlite3.IntegrityError:
            pass

    # Add courses
    faculty_ids = [row[0] for row in cursor.execute("SELECT id FROM faculty").fetchall()]
    for _ in range(200):
        cursor.execute("INSERT INTO courses (name, faculty_id) VALUES (?, ?)", 
                       (fake.bs(), fake.random_element(elements=faculty_ids)))

    # Add enrollments
    student_ids = [row[0] for row in cursor.execute("SELECT id FROM students").fetchall()]
    course_ids = [row[0] for row in cursor.execute("SELECT id FROM courses").fetchall()]
    for _ in range(1500):
        try:
            cursor.execute("INSERT INTO enrollments (student_id, course_id) VALUES (?, ?)",
                           (fake.random_element(elements=student_ids), fake.random_element(elements=course_ids)))
        except sqlite3.IntegrityError:
            pass

    conn.commit()
    conn.close()
    print("Database setup complete. 'university.db' created and populated.")

if __name__ == "__main__":
    setup_database()
