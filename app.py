import streamlit as st
import os
import pandas as pd
import glob
import base64
from datetime import datetime
from gtts import gTTS
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ==========================================
# 1. KONFIGURASI & API
# ==========================================
st.set_page_config(page_title="AI PDF Assistant Pro", page_icon="🎙️", layout="wide")

if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
else:
    st.error("⚠️ Masukkan GROQ_API_KEY di Secrets!")
    st.stop()

# ==========================================
# 2. FUNGSI AUDIO (FIXED)
# ==========================================
def get_audio_html(text, lang='id'):
    """Mengonversi teks ke base64 audio agar bisa diputar berulang kali."""
    try:
        tts = gTTS(text=text, lang=lang)
        filename = "temp_audio.mp3"
        tts.save(filename)
        
        with open(filename, "rb") as f:
            data = f.read()
            b64 = base64.b64encode(data).decode()
            
        # Hapus file fisik setelah diubah ke base64 agar hemat storage
        if os.path.exists(filename):
            os.remove(filename)
            
        return f"""
            <audio controls>
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            Your browser does not support the audio element.
            </audio>
            """
    except Exception as e:
        return f"Error Audio: {e}"

def clear_audio_cache():
    """Membersihkan file mp3 sisa jika ada."""
    for f in glob.glob("*.mp3"):
        os.remove(f)

# ==========================================
# 3. FUNGSI RAG & LOGGING
# ==========================================
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def process_pdf(uploaded_file):
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    loader = PyPDFLoader("temp.pdf")
    pages = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_documents(pages)
    return Chroma.from_documents(documents=chunks, embedding=load_embeddings())

def save_feedback(query, response, rating, pages):
    df = pd.DataFrame([{
        "time": datetime.now(), "query": query, "rating": "Positive" if rating==1 else "Negative"
    }])
    df.to_csv("logs_evaluasi.csv", mode='a', header=not os.path.exists("logs_evaluasi.csv"), index=False)

# ==========================================
# 4. UI & SESSION STATE
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_audio" not in st.session_state:
    st.session_state.current_audio = None

with st.sidebar:
    st.header("📁 Dokumen")
    pdf_file = st.file_uploader("Upload PDF", type="pdf")
    if st.button("🚀 Proses"):
        if pdf_file:
            st.session_state.vectorstore = process_pdf(pdf_file)
            st.success("Berhasil!")
    
    st.divider()
    if st.button("🧹 Clear Audio Cache"):
        clear_audio_cache()
        st.toast("Cache dibersihkan")
    
    if st.button("🗑️ Reset Chat"):
        st.session_state.messages = []
        st.session_state.current_audio = None
        st.rerun()

# Display Chat
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# ==========================================
# 5. LOGIKA UTAMA (FIX AUDIO PERSISTENCE)
# ==========================================
if prompt := st.chat_input("Tanya sesuatu..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if "vectorstore" in st.session_state:
        with st.chat_message("assistant"):
            with st.spinner("Berpikir..."):
                # Retrieval & LLM
                results = st.session_state.vectorstore.similarity_search(prompt, k=3)
                context = "\n".join([d.page_content for d in results])
                pages = [d.metadata.get('page', 0)+1 for d in results]
                
                llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.1)
                ans = llm.invoke(f"Context: {context}\nQuestion: {prompt}").content
                
                full_res = f"{ans}\n\n> 📍 Halaman: {list(set(pages))}"
                st.markdown(full_res)
                
                # Generate Audio & Simpan ke Session State
                lang = 'en' if any(x in prompt.lower() for x in ['what', 'how', 'tell']) else 'id'
                st.session_state.current_audio = get_audio_html(ans, lang)
                
                # Simpan metadata untuk feedback
                st.session_state.last_query = prompt
                st.session_state.last_answer = full_res
                st.session_state.last_pages = pages
                st.session_state.messages.append({"role": "assistant", "content": full_res})
                st.rerun()

# Widget Audio (Ditampilkan di luar chat input agar tetap persisten)
if st.session_state.current_audio:
    st.divider()
    st.write("🔊 **Dengarkan Jawaban Terakhir:**")
    st.markdown(st.session_state.current_audio, unsafe_allow_html=True)
    st.caption("Klik tombol play di atas. Anda bisa memutarnya berkali-kali.")

# Feedback Section
if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
    feedback = st.feedback("thumbs")
    if feedback is not None:
        save_feedback(st.session_state.last_query, st.session_state.last_answer, feedback, st.session_state.last_pages)
        st.toast("Feedback disimpan!")
