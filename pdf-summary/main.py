import PyPDF2
import ollama
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever

hf_model_name = "sentence-transformers/all-mpnet-base-v2"
embedding_model = HuggingFaceEmbeddings(model_name=hf_model_name)

vector_stores = {
    "general": Chroma(collection_name="general", embedding_function=embedding_model)
}
bm25_retrievers = {}
summaries = []

def get_summary(chunk,model="deepseek-r1:7b"):
    try:
        response = ollama.chat(
            model=model,
            messages=[{
                "role": "user",
                "content": f"""
You are an expert summarization assistant. Summarize the following text in detail.

### TEXT TO SUMMARIZE ###
{chunk}
"""
            }]
        )
        return response["message"]["content"]
    except Exception as e:
        print(f"Error calling Ollama API: {e}")
        return ""
    

if __name__ == "__main__":
    pdf_path = "DD.pdf"
    pdf_reader = PyPDF2.PdfReader(pdf_path)
    full_text = ""
    for page in pdf_reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n"
    
    if not full_text.strip():
        print("Unscannnable pdf")

    file_name = pdf_path.strip("/")[-1]

    text_spliter = RecursiveCharacterTextSplitter(
        chunk_size=5000,
        chunk_overlap=700,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_spliter.split_text(full_text)
    
    summary_texts = []
    
    for chunk in chunks:
        s = get_summary(chunk)
        if s:
            summary_texts.append(s)
        final_summary = "\n\n".join(summary_texts)
        summaries.insert(0,{"file_name":pdf_path, "summary":final_summary})
        print(f"PDF {pdf_path} processed and summarized")

    for entry in summaries:
        print("\##########")
        print(f"Summary for {entry['file_name']}:\n{entry['summary']}")
