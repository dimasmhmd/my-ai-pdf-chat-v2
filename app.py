import streamlit as st
import os
import pandas as pd
import glob
from datetime import datetime
from gtts import gTTS
import base64
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ==========================================
# 1. KONFIGURASI HALAMAN & API
# ==========================================
st.set_page_config(page_title="AI PDF Assistant Ultimate", page_icon="🎙️", layout="wide")

if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
else:
    st.error("⚠️ GROQ_API_KEY tidak ditemukan di Secrets!")
    st.stop()

# ==========================================
# 2. FUNGSI UTILITAS (TTS, CACHE, LOGGING)
# ==========================================
def clear_audio_cache():
    """Menghapus semua file mp3 di direktori kerja."""
    files = glob.glob("*.mp3")
    for f in files:
        try:
            os.remove(f)
        except:
            pass
    return len(files)

def text_to_speech(text, lang='id'):
    """Mengonversi teks ke suara dengan nama file unik berbasis timestamp."""
    # Bersihkan cache lama sebelum membuat yang baru
    clear_audio_cache()
    
    filename = f"res_{datetime.now().strftime('%H%M%S')}.mp3"
    tts = gTTS(text=text, lang=lang)
    tts.save(filename)
    
    with open(filename, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        md = f"""
            <audio controls autoplay="true">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """
        return md

def save_feedback(user_query, ai_response, rating, pages):
    log_file = "logs_evaluasi.csv"
    feedback_data = {
        "timestamp": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        "query": [user_query],
        "response": [ai_response.replace('\n', ' ')],
        "pages": [str(pages)],
        "rating": ["Positive" if rating == 1 else "Negative"]
    }
    df_new = pd.DataFrame(feedback_data)
    if not os.path.isfile(log_file):
        df_new.to_csv(log_file, index=False)
    else:
        df_new.to_csv(log_file, mode='a', header=False, index=False)

# ==========================================
# 3. FUNGSI RAG
# ==========================================
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def process_pdf(uploaded_file):
    try:
        # Bersihkan audio saat proses dokumen baru
        clear_audio_cache()
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        loader = PyPDFLoader("temp.pdf")
        pages = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
        chunks = text_splitter.split_documents(pages)
        vectorstore = Chroma.from_documents(
            documents=chunks, embedding=load_embeddings(), collection_name="ultimate_rag_db"
        )
        return vectorstore
    except Exception as e:
        st.error(f"Error: {e}")
        return None

# ==========================================
# 4. ANTARMUKA PENGGUNA (UI)
# ==========================================
st.title("🎙️ AI PDF Assistant: Voice & Cache Manager")

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- SIDEBAR ---
with st.sidebar:
    st.header("📁 Manajemen Dokumen")
    pdf_file = st.file_uploader("Unggah PDF", type="pdf")
    if st.button("🚀 Proses"):
        if pdf_file:
            with st.spinner("Menganalisis..."):
                st.session_state.vectorstore = process_pdf(pdf_file)
                st.success("Selesai!")

    st.divider()
    st.header("💾 Chat & Server Management")
    
    # Fitur Download History
    if st.session_state.messages:
        chat_text = "".join([f"{m['role'].upper()}: {m['content']}\n\n" for m in st.session_state.messages])
        st.download_button("📥 Download Chat (.txt)", chat_text, f"chat_{datetime.now().strftime('%Y%m%d')}.txt")

    # Tombol Manual Clear Cache
    if st.button("🧹 Clear Audio Cache"):
        count = clear_audio_cache()
        st.toast(f"Berhasil menghapus {count} file audio sampah.", icon="🗑️")

    if st.button("🗑️ Hapus Chat"):
        st.session_state.messages = []
        clear_audio_cache()
        st.rerun()

# Display Chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ==========================================
# 5. LOGIKA TANYA JAWAB & TTS
# ==========================================
if prompt := st.chat_input("Apa yang ingin Anda tanyakan?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if "vectorstore" in st.session_state:
        with st.chat_message("assistant"):
            with st.spinner("Mencari referensi..."):
                search_results = st.session_state.vectorstore.similarity_search(prompt, k=4)
                context = "\n\n".join([d.page_content for d in search_results])
                pages = sorted(list(set([d.metadata.get('page', 0) + 1 for d in search_results])))
                
                llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.1)
                sys_prompt = f"Anda asisten PDF. Jawab sesuai bahasa user.\n\nKONTEKS:\n{context}"
                response = llm.invoke(sys_prompt + f"\n\nPERTANYAAN: {prompt}")
                answer = response.content
                
                full_res = f"{answer}\n\n> 📍 Referensi: Halaman {', '.join(map(str, pages))}"
                st.markdown(full_res)
                
                # TTS Section
                with st.expander("🔊 Dengarkan Jawaban"):
                    audio_lang = 'en' if any(word in prompt.lower()[:10] for word in ['what', 'how', 'is', 'can']) else 'id'
                    audio_html = text_to_speech(answer, lang=audio_lang)
                    st.markdown(audio_html, unsafe_allow_html=True)

                # Source Preview
                with st.expander("🔍 Lihat Teks Asli"):
                    for i, doc in enumerate(search_results):
                        st.info(f"**Hal {doc.metadata.get('page', 0)+1}:** {doc.page_content}")

                st.session_state.last_query = prompt
                st.session_state.last_answer = full_res
                st.session_state.last_pages = pages
                st.session_state.messages.append({"role": "assistant", "content": full_res})
                st.rerun()
    else:
        st.info("💡 Upload PDF dulu.")

# Feedback Section
if "messages" in st.session_state and len(st.session_state.messages) > 0:
    if st.session_state.messages[-1]["role"] == "assistant":
        st.write("---")
        feedback = st.feedback("thumbs")
        if feedback is not None:
            save_feedback(st.session_state.last_query, st.session_state.last_answer, feedback, st.session_state.last_pages)
            st.toast("Feedback tersimpan!")
