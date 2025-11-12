---
title: RAG Pipeline Demo
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: "1.37.0"
app_file: app.py
pinned: false
---

# 🎓 University Knowledge Retrieval System

A comprehensive Retrieval-Augmented Generation (RAG) system for university data with advanced guardrails and experimental validation.

## 🌟 Features

- **Interactive Chat Interface**: Natural language queries about university data
- **Advanced Guardrails**: Enhanced input/output security, including blocking queries with Austrian social security numbers (SVNRs) and redacting them from responses.
- **Experimental Dashboard**: Comprehensive testing suite for RAG validation.
- **Real Database**: 6,000+ students, 1,300+ faculty, 2,600+ courses, with student records now including SVNRs for guardrail testing.
- **Vector Search**: Semantic search using ChromaDB and Sentence Transformers, with SVNRs intentionally included in student documents for output guardrail testing.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Hugging Face API Token

### Installation

1. **Clone and setup**:
```bash
git clone <repository-url>
cd projekt
pip install -r requirements.txt
```

2. **Configure API Key** (choose one):
   - Create `secrets_local.py`: `HF = "your_hugging_face_token"`
   - Or set environment variable: `HF_TOKEN=your_token`

3. **Initialize Database and Vector Store**:
   Run the setup scripts to create the SQLite database and populate the vector store.
```bash
python database/setup_db.py
python rag/build_vector_store.py
```

4. **Run the application**:
```bash
streamlit run app.py
```

## 📊 System Components

### Chat Interface
- Natural language queries about university data
- Real-time RAG pipeline with source citations
- Input/output guardrails for security

### Experimental Dashboard
Two comprehensive test suites:
1. **Input Guardrails**: Tests against malicious inputs (SQL injection, PII extraction)
2. **Output Guardrails**: Validates response quality and detects hallucinations

## 🏗️ Architecture

```
├── app.py                    # Main Streamlit application
├── experimental_dashboard.py # Experiment interface and system info
├── experiments/             # Test suites for RAG validation
│   ├── experiment_1_input_guardrails.py
│   ├── experiment_2_output_guardrails.py
│   ├── experiment_3_hyperparameters.py
│   └── experiment_4_context_window.py
├── database/               # SQLite university database
├── rag/                   # Vector store and retrieval
├── rails/                 # Input/output guardrails
├── model/                 # RAG model integration
└── guards/               # Security components
```

## 🔧 Configuration

**Dependencies** (requirements.txt):
- streamlit==1.37.0
- sentence-transformers==5.1.0
- chromadb==1.0.21
- Faker==15.3.4 (for database generation)
- huggingface-hub==0.34.4
- nltk, numpy, scikit-learn

## 🎯 Usage Examples

**Student Queries**:
- "What courses is Maria taking?"
- "Who are the students in computer science?"

**Faculty Queries**:
- "Who teaches in the engineering department?"
- "Show me all professors"

**Course Queries**:
- "What courses are available?"
- "Who teaches advanced mathematics?"

## 🧪 Running Experiments

Access via the "Experiments" tab in the web interface, or run individually:

```bash
cd experiments
python experiment_1_input_guardrails.py
python experiment_2_output_guardrails.py
```

## 🔒 Security Features

- **Input Validation**: SQL injection prevention, malicious prompt detection, **blocking queries containing valid Austrian social security numbers (SVNRs)**.
- **Output Filtering**: PII redaction (including **SVNRs**), hallucination detection, relevance checking.
- **Content Sanitization**: Automatic cleaning of responses and database content.

## 📈 Database Statistics

- **Students**: 6,398 records with realistic personal data
- **Faculty**: 1,297 professors across multiple departments
- **Courses**: 2,600 courses linked to faculty
- **Enrollments**: 19,443 student-course relationships

## 🔑 API Requirements

Requires Hugging Face API access for:
- Text generation models
- Embedding models for semantic search
- Guardrail validation services
