import streamlit as st
import os
import json

# --- GEREKLİ KÜTÜPHANELER ---
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# --- 1. AYARLAR ---
if validate_secrets := getattr(st, "secrets", None):
    if "GOOGLE_API_KEY" in st.secrets:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

if "GOOGLE_API_KEY" not in os.environ:
    st.error("Google API Key bulunamadı! Lütfen Streamlit Secrets (veya .env) içine GOOGLE_API_KEY ekleyin.")
    st.stop()

st.set_page_config(page_title="METU-IE Bot", page_icon="🤖")
st.title("METU-IE Staj Danışmanı")

# --- 2. RAG PIPELINE KURULUMU ---
@st.cache_resource
def setup_rag_pipeline():
    db_directory = "./chroma_db"
    if not os.path.exists(db_directory):
        st.error(f"'{db_directory}' klasörü bulunamadı! Lütfen önce veritabanını oluşturan dosyayı çalıştırın.")
        return None
    try:
        # 404 HATASINI BİTİREN YEREL VE HAFİF MODEL (Sadece 45MB)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # Hazır veritabanını chroma_db klasöründen yükle
        vectorstore = Chroma(persist_directory=db_directory, embedding_function=embeddings)
        
        # Soru başına 3 kaynak dönecek retriever
        retriever = vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 3, "score_threshold": 0.4}
        )

        # LLM Tanımlaması
        llm = ChatGoogleGenerativeAI(
            model="gemini-3.1-flash-lite-preview",
            temperature=0.1,
            convert_system_message_to_human=True
        )

        system_prompt = (
            "You are the METU Industrial Engineering Summer Practice Consultant. "
            "Your expertise is strictly limited to IE 300 and IE 400 internships. "
            "Use the following pieces of retrieved context to answer the question. "
            "Do NOT reject short or specific internship questions (e.g., 'duration'), "
            "if the context contains the answer, provide it concisely. "
            "CRITICAL RULE: Before answering, check if the user's question is actually about internships. If the "
            "question is unrelated such as life advice, general chatter, random words or non-IE topics "
            "(e.g., 'how are you', 'what are you doing'), you MUST ignore the retrieved context "
            "and politely state that you only handle summer practice queries. If the user's input "
            "is a random word, greeting, or gibberish (e.g., 'tasty', 'asdf', 'hello') that has no "
            "relation to internships, do NOT use the context. Simply say: 'I am the METU-IE Internship Bot. Please "
            "ask a specific question about IE 300 or IE 400.'"
            "If the question is close in context but too vague to give a concise answer, "
            "politely ask the user to ask again. "
            "If the user's question mentions IE300 or IE400, you must only answer using context that matches that exact internship code. Never answer an IE400 question with IE300 information, and never answer an IE300 question with IE400 information."
            "If the retrieved context already contains the answer, especially from 'Custom FAQ Dataset', answer directly and do not redirect the user to the website."
            "Always be helpful, concise, and professional. Be student-friendly.\n\n"
            "{context}"
        )
        
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )

        contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])

        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_q_prompt
        )

        qa_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])

        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        
        return rag_chain
    except Exception as e:
        st.error(f"Pipeline yükleme hatası: {e}")
        return None

# --- 3. SİSTEMİ ÇALIŞTIR ---
chatbot = setup_rag_pipeline()

# Pipeline başarıyla kurulduysa devam et
if chatbot is not None:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Sorunu sor..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Yanıt hazırlanıyor..."):
                chat_history = []
                for msg in st.session_state.messages[:-1]: # exclude the latest prompt
                    if msg["role"] == "user":
                        chat_history.append(HumanMessage(content=msg["content"]))
                    else:
                        chat_history.append(AIMessage(content=msg["content"]))

                # Pass chat_history to the chain
                response = chatbot.invoke({
                    "input": prompt,
                    "chat_history": chat_history
                })
                answer = response["answer"]
                
                # Kaynakları da mesaja ekleyelim
                sources = response.get("context", [])
                if sources:
                    answer += "\n\n**Kaynaklar:**\n"
                    used_sources = set()
                    for doc in sources:
                        src = doc.metadata.get("source", "Bilinmeyen Kaynak")
                        if src not in used_sources:
                            used_sources.add(src)
                            answer += f"- {src}\n"

                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
else:
    st.warning("Pipeline (Database ve Model) yüklenemediği için sohbet başlatılamıyor. Lütfen terminaldeki hatalara bakın.")