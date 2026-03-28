import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 1. Set up your Gemini API Key
# Replace this with your actual key from Google AI Studio
os.environ["GOOGLE_API_KEY"] = "AIzaSyBqjXBqkp2brM0zFbwlO6DIYnWPa3GqSBs"

def setup_rag_pipeline(db_dir):
    # 2. Re-load the embedding model and the local database
    print("Loading database...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory=db_dir, embedding_function=embeddings)
    
    # Create a retriever that fetches the top 3 most relevant text chunks
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 3. Initialize the Gemini LLM
    print("Initializing Gemini LLM...")
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", # Fast and highly efficient for RAG tasks
        temperature=0.1 # Keep this very low so the bot stays factual and doesn't hallucinate
    )

    # 4. Create the System Prompt
    # This acts as the strict instruction manual for your chatbot
    system_prompt = (
        "You are an intelligent virtual consultant for METU Industrial Engineering students. "
        "Use the following pieces of retrieved context to answer the question. "
        "If the answer is not in the provided context, gracefully state that you do not know "
        "and advise the student to check the official sp-ie.metu.edu.tr website. "
        "Always be helpful, concise, and professional.\n\n"
        "{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # 5. Build the Final Chain
    # This combines the retriever (database search) and the LLM (Gemini) into one workflow
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain

if __name__ == "__main__":
    db_directory = "./chroma_db"
    
    # Initialize the pipeline
    chatbot = setup_rag_pipeline(db_directory)
    
    # Let's test it with a sample query from your project instructions
    print("\n--- METU IE Summer Practice Bot ---")
    question = "Which programming languages are recommended?"
    print(f"User: {question}")
    
    # Invoke the chain
    response = chatbot.invoke({"input": question})
    
    print(f"\nBot: {response['answer']}")
    
    # Print the sources so you can verify Contextual Integrity
    print("\nSources Used:")
    for doc in response['context']:
        print(f"- {doc.metadata.get('source', 'Unknown URL')}")