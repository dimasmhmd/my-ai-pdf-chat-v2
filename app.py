import streamlit as st
import os
import pandas as pd
import glob
import base64
from datetime import datetime
from gtts import gTTS
from langdetect import detect
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ==========================================
# 1. KONFIGURASI HALAMAN & API
# ==========================================
st.set_page_config(page_title="Ultimate AI PDF Assistant", page_icon="🎙️", layout="wide")

# Mengambil API Key dari Streamlit Secrets
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
else:
    st.error("⚠️ GROQ_API_KEY tidak ditemukan di Secrets!")
    st.stop()

# ==========================================
# 2. FUNGSI AUDIO & DETEKSI AKSEN
# ==========================================
def get_audio_html(text):
    """Mengonversi teks ke audio dengan deteksi bahasa otomatis untuk aksen yang tepat."""
    try:
        # Deteksi bahasa dari teks jawaban
        try:
            detected_lang = detect(text)
            # Batasi ke bahasa yang umum didukung gTTS
            if detected_lang not in ['id', 'en', 'ja', 'ko', 'fr', 'de']:
                detected_lang = 'id'
        except:
            detected_lang = 'id'

        tts = gTTS(text=text, lang=detected_lang)
        filename = f"temp_voice_{detected_lang}.mp3"
        tts.save(filename)
        
        with open(filename, "rb") as f:
            data = f.read()
            b64 = base64.b64encode(data).decode()
            
        # Hapus file fisik segera setelah di-encode ke base64
        if os.path.exists(filename):
            os.remove(filename)
            
        return f"""
            <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; border-left: 5px solid #ff4b4b; margin-top: 10px;">
                <p style="margin-bottom: 5px; font-weight: bold; color: #31333F;">🔊 Pemutar Suara (Aksen: {detected_lang.upper()}):</p>
                <audio controls style="width: 100%;">
                    <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
                </audio>
            </div>
            """, detected_lang
    except Exception as e:
        return f"Error Audio: {e}", "id"

def clear_audio_cache():
    """Membersihkan file mp3 yang mungkin tertinggal."""
    files = glob.glob("*.mp3")
    for f in files:
        os.remove(f)
    return len(files)

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
    log_file = "logs_evaluasi.csv"
    df = pd.DataFrame([{
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "query": query,
        "rating": "Positive" if rating == 1 else "Negative",
        "pages": str(pages)
    }])
    df.to_csv(log_file, mode='a', header=not os.path.exists(log_file), index=False)

# ==========================================
# 4. INITIALIZATION & SIDEBAR
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_audio" not in st.session_state:
    st.session_state.current_audio = None

with st.sidebar:
    st.header("📁 File Manager")
    pdf_file = st.file_uploader("Upload PDF Dokumen", type="pdf")
    if st.button("🚀 Proses & Analisis"):
        if pdf_file:
            with st.spinner("Sedang memproses dokumen..."):
                st.session_state.vectorstore = process_pdf(pdf_file)
                st.success("Analisis PDF Berhasil!")
    
    st.divider()
    st.header("⚙️ Server Tools")
    if st.button("🧹 Clear Audio Cache"):
        count = clear_audio_cache()
        st.toast(f"Dihapus {count} file audio.", icon="🗑️")
    
    if st.button("🗑️ Reset Chat"):
        st.session_state.messages = []
        st.session_state.current_audio = None
        st.rerun()

# ==========================================
# 5. LOGIKA CHAT UTAMA
# ==========================================
st.title("🎙️ AI PDF Assistant Pro")
st.markdown("Tanya jawab dengan PDF menggunakan suara dan deteksi bahasa otomatis.")

# Tampilkan history chat
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Input user
if prompt := st.chat_input("Tanyakan sesuatu tentang isi PDF..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if "vectorstore" in st.session_state:
        with st.chat_message("assistant"):
            with st.spinner("Mencari referensi..."):
                # A. Retrieval
                results = st.session_state.vectorstore.similarity_search(prompt, k=3)
                context = "\n".join([d.page_content for d in results])
                pages = sorted(list(set([d.metadata.get('page', 0) + 1 for d in results])))
                
                # B. Generation (LLM)
                llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.1)
                sys_prompt = f"""
                Detect language and answer in the same language as the user.
                Context from PDF: {context}
                """
                response = llm.invoke(sys_prompt + f"\n\nQuestion: {prompt}")
                ans = response.content
                
                # C. Final Format
                full_res = f"{ans}\n\n> 📍 **Referensi:** Halaman {', '.join(map(str, pages))}"
                st.markdown(full_res)
                
                # D. Generate Audio (Persistent)
                audio_html, lang_code = get_audio_html(ans)
                st.session_state.current_audio = audio_html
                
                # E. Source Preview
                with st.expander("🔍 Lihat Potongan Teks Asli"):
                    for i, doc in enumerate(results):
                        st.info(f"**Hal {doc.metadata.get('page',0)+1}:** {doc.page_content}")

                # Save Metadata
                st.session_state.last_query = prompt
                st.session_state.last_answer = full_res
                st.session_state.last_pages = pages
                st.session_state.messages.append({"role": "assistant", "content": full_res})
                st.rerun()
    else:
        st.info("💡 Silakan unggah PDF terlebih dahulu di sidebar.")

# ==========================================
# 6. PERSISTENT AUDIO & FEEDBACK (BOTTOM)
# ==========================================
if st.session_state.current_audio:
    st.write("---")
    st.markdown(st.session_state.current_audio, unsafe_allow_html=True)

if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
    st.caption("Apakah jawaban ini membantu?")
    feedback = st.feedback("thumbs")
    if feedback is not None:
        save_feedback(st.session_state.last_query, st.session_state.last_answer, feedback, st.session_state.last_pages)
        st.toast("Terima kasih atas feedback Anda!", icon="⭐")
