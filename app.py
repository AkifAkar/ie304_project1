import streamlit as st
import os
import json

# --- GEREKLİ KÜTÜPHANELER ---
from langchain_huggingface import HuggingFaceEmbeddings # Yerel ve hafif model için
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- 1. AYARLAR ---
os.environ["GOOGLE_API_KEY"] = ""

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
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        # LLM Tanımlaması
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.1
        )

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

        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        
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
                response = chatbot.invoke({"input": prompt})
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