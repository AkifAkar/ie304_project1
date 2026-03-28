import os
import json
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def load_custom_faqs(json_filepath):
    print(f"Loading custom FAQs from {json_filepath}...")
    faq_documents = []
    
    try:
        with open(json_filepath, 'r', encoding='utf-8') as f:
            faq_data = json.load(f)
            
        for item in faq_data:
            # Combine the Q and A into one searchable chunk
            combined_text = f"Question: {item['question']}\nAnswer: {item['answer']}"
            
            # Create the Document and tag it so the bot knows the source
            doc = Document(
                page_content=combined_text,
                metadata={"source": "Custom FAQ Dataset"}
            )
            faq_documents.append(doc)
            
        print(f"  Successfully loaded {len(faq_documents)} custom FAQs.")
        
    except FileNotFoundError:
        print(f"  Warning: {json_filepath} not found. Skipping FAQs.")
    except json.JSONDecodeError:
        print(f"  Error: {json_filepath} is not formatted correctly.")
        
    return faq_documents

def build_vector_database(input_filename, faq_filename, persist_dir):
    print("1. Reading the cleaned text file...")
    with open(input_filename, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split the file back into our individual chunks
    raw_chunks = content.split("--- CHUNK ")
    
    documents = []
    
    print("2. Parsing chunks into LangChain Documents...")
    for chunk in raw_chunks:
        if not chunk.strip():
            continue
            
        lines = chunk.strip().split('\n')
        
        source_url = "Unknown Source"
        text_content = []
        
        # Go through each line to separate the URL from the actual text
        for line in lines:
            if line.startswith("SOURCE: "):
                source_url = line.replace("SOURCE: ", "").strip()
            elif line.startswith("CONTENT: "):
                text_content.append(line.replace("CONTENT: ", "").strip())
            elif not line.startswith(tuple("0123456789")): # Ignore the chunk number line
                text_content.append(line.strip())
                
        # Join the text back together
        full_text = " ".join(text_content).strip()
        
        if full_text:
            # Create a Document object. Metadata stores the URL for citations later!
            doc = Document(
                page_content=full_text,
                metadata={"source": source_url}
            )
            documents.append(doc)

    print(f"Successfully parsed {len(documents)} website documents.")

    print("3. Processing Custom FAQs...")
    # Load the FAQs and add them to our main list of documents
    faq_documents = load_custom_faqs(faq_filename)
    documents.extend(faq_documents) 
    print(f"Total documents to embed: {len(documents)}")

    print("4. Downloading Embedding Model (this may take a minute the first time)...")
    # This translates our text into numerical vectors
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print("5. Creating Chroma Vector Database...")
    # This creates the database and saves it to a folder on your computer
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    
    print(f"Success! Database saved to the '{persist_dir}' directory.")
    return vectorstore

if __name__ == "__main__":
    input_file = "clean_metu_sp_chunks.txt"
    faq_file = "custom_faqs.json" 
    db_directory = "./chroma_db"
    
    build_vector_database(input_file, faq_file, db_directory)