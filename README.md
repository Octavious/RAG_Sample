<a href="https://www.buymeacoffee.com/ArabicAI" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" style="height: 60px !important;width: 217px !important;" ></a>
# RAG System with Ollama and FAISS

A Retrieval-Augmented Generation (RAG) system that uses Ollama for embeddings and language model, and FAISS for vector storage.

## Features

- Load and process PDF and Excel files
- Generate embeddings using Ollama's all-minilm model
- Store vectors using FAISS
- Query both structured (Excel) and unstructured (PDF) data
- Detailed employee vacation tracking
- Company policy integration

## Prerequisites

- Python 3.9+
- Ollama installed and running locally with the following models:
  - all-minilm (for embeddings)
  - llama3.2 (for text generation)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Octavious/RAG_Sample.git
cd RAG_Sample
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Make sure Ollama is running and the required models are installed:
```bash
ollama pull all-minilm
ollama pull llama2
```

## Usage

1. Place your documents in the project directory:
   - PDF files (e.g., CompanyVacationPolicy.pdf)
   - Excel files (e.g., employee_list.xlsx)

2. Run the RAG system:
```bash
python rag_system.py
```

3. Ask questions about:
   - Employee vacation days
   - Company vacation policies
   - Department information
   - Employee records

## Example Questions

- "How many vacation days has [employee name] taken?"
- "What is the vacation policy for maternity leave?"
- "Who works in the HR department?"
- "How many vacation days remain for [employee name]?"
