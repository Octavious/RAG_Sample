import os
from langchain_community.document_loaders import PyPDFLoader, UnstructuredExcelLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain_community.docstore.document import Document
import requests.exceptions
import pandas as pd

def load_excel_file(file_path):
    """Load Excel file and convert to structured text documents"""
    print(f"\nAttempting to load Excel file: {file_path}")
    
    try:
        # Read Excel file with pandas
        df = pd.read_excel(file_path)
        print(f"Excel file structure:")
        print(f"- Number of rows: {len(df)}")
        print(f"- Columns: {', '.join(df.columns)}")
        
        # Convert each row to a well-formatted text document
        documents = []
        for _, row in df.iterrows():
            # Create a natural language description of the row
            content = f"Employee Record:\n"
            content += f"Name: {row['Name']}\n"
            content += f"Position: {row['Position']}\n"
            content += f"Department: {row['Department']}\n"
            content += f"Gender: {row['Gender']}\n"
            content += f"Vacation Days Taken: {row['Vacation Taken']}\n"
            content += f"Sick Leaves Taken: {row['Sick Leaves']}\n"
            
            # Create a Document with metadata
            doc = Document(
                page_content=content,
                metadata={
                    "source": file_path,
                    "row_index": _,
                    "employee_name": row['Name']
                }
            )
            documents.append(doc)
            
        print(f"Successfully created {len(documents)} documents from Excel rows")
        return documents
    except Exception as e:
        print(f"Error processing Excel file: {str(e)}")
        return []

# Initialize Ollama embeddings with the all-minilm model
embeddings = OllamaEmbeddings(model="all-minilm")

# Initialize document loaders
print("Loading documents...")
pdf_loader = PyPDFLoader("CompanyVacationPolicy.pdf")

# Load documents
try:
    pdf_docs = pdf_loader.load()
    print("PDF loaded successfully")
except Exception as e:
    print(f"Error loading PDF: {str(e)}")
    pdf_docs = []

excel_docs = load_excel_file("employee_list.xlsx")

documents = pdf_docs + excel_docs

if not documents:
    print("No documents were loaded. Please check if the files exist and are accessible.")
    exit(1)

print(f"\nTotal documents loaded: {len(documents)}")
print("\nDocument sources:")
for doc in documents:
    print(f"- Source: {doc.metadata.get('source', 'Unknown')}")
    if 'employee_name' in doc.metadata:
        print(f"  Employee: {doc.metadata['employee_name']}")
    print(f"  Content preview: {doc.page_content[:100]}...")

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)
splits = text_splitter.split_documents(documents)
print(f"\nCreated {len(splits)} text chunks")

# Create vector store using FAISS
print("Creating vector store...")
vectorstore = FAISS.from_documents(
    documents=splits,
    embedding=embeddings
)

# Save the vector store
vectorstore.save_local("faiss_index")
print("Vector store created and saved")

# Initialize Ollama LLM
llm = OllamaLLM(model="llama3.2")

# Create prompt template
template = """Answer the following question based on the provided context. Follow these rules:
1. If the question is about vacation days, ALWAYS mention the company policy of 25 paid vacation days per year
2. For questions about specific employees, clearly state:
   - Their position and department
   - Number of vacation days taken
   - Number of vacation days remaining (25 minus days taken)
3. If looking at historical data, mention when we don't have the full year's context

Context: {context}

Question: {question}

Answer the question using ONLY the information from the context. If you cannot answer this question based on the context, say "I cannot answer this question based on the available information."

Answer:"""

prompt = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)

# Create retrieval chain
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt 
    | llm 
    | StrOutputParser()
)

def query_documents(question: str):
    """
    Query the RAG system with a question
    """
    try:
        return rag_chain.invoke(question)
    except requests.exceptions.ConnectionError:
        return "Error: Could not connect to Ollama. Please make sure the Ollama service is running."
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    print("\nRAG System initialized. You can ask questions about the company vacation policy and employee list.")
    print("Type 'quit' to exit.")
    print("Make sure Ollama service is running before asking questions.")
    print("\nExample questions you can ask:")
    print("- How many vacation days has [employee name] taken?")
    print("- What is the vacation policy for maternity leave?")
    print("- Who works in the HR department?")
    
    while True:
        question = input("\nEnter your question: ")
        if question.lower() == 'quit':
            break
        if not question.strip():
            continue
            
        try:
            print("\nSearching for answer...")
            response = query_documents(question)
            print("\nAnswer:", response)
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {str(e)}")