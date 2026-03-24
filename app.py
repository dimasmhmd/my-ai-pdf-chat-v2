import streamlit as st
import os
import pandas as pd
from datetime import datetime
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ==========================================
# 1. KONFIGURASI & API KEY
# ==========================================
st.set_page_config(page_title="AI PDF Pro + Monitoring", page_icon="🤖", layout="wide")

if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
else:
    st.error("⚠️ GROQ_API_KEY tidak ditemukan di Secrets!")
    st.stop()

# ==========================================
# 2. FUNGSI LOGGING & EVALUASI (LLMOps)
# ==========================================
def save_feedback(user_query, ai_response, rating, pages):
    """Menyimpan feedback pengguna ke file CSV lokal."""
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
# 3. FUNGSI RAG (PROSES DOKUMEN)
# ==========================================
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def process_pdf(uploaded_file):
    try:
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        loader = PyPDFLoader("temp.pdf")
        pages = loader.load()
        
        # Splitter Teroptimasi
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200, chunk_overlap=200, 
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = text_splitter.split_documents(pages)
        
        vectorstore = Chroma.from_documents(
            documents=chunks, 
            embedding=load_embeddings(),
            collection_name="pdf_log_db"
        )
        return vectorstore
    except Exception as e:
        st.error(f"Gagal proses PDF: {e}")
        return None

# ==========================================
# 4. ANTARMUKA PENGGUNA (UI)
# ==========================================
st.title("📄 Smart PDF Explorer & Auditor")
st.markdown("Sistem RAG Teroptimasi dengan Fitur Monitoring Evaluasi.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.header("📁 Unggah & Kontrol")
    pdf_file = st.file_uploader("Pilih PDF", type="pdf")
    
    if st.button("🚀 Proses Dokumen"):
        if pdf_file:
            with st.spinner("Menganalisis..."):
                st.session_state.vectorstore = process_pdf(pdf_file)
                st.success("Analisis Selesai!")
        else:
            st.warning("Pilih file dulu.")
    
    st.divider()
    if st.button("🗑️ Hapus Chat"):
        st.session_state.messages = []
        st.rerun()
    
    # Statistik Singkat Feedback
    if os.path.exists("logs_evaluasi.csv"):
        st.header("📊 Statistik Feedback")
        df_log = pd.read_csv("logs_evaluasi.csv")
        st.write(df_log['rating'].value_counts())

# Tampilkan Chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ==========================================
# 5. LOGIKA TANYA JAWAB & FEEDBACK
# ==========================================
if prompt := st.chat_input("Tanyakan isi PDF..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if "vectorstore" in st.session_state:
        with st.chat_message("assistant"):
            with st.spinner("Mencari referensi..."):
                # A. Retrieval
                search_results = st.session_state.vectorstore.similarity_search(prompt, k=4)
                context = "\n\n".join([d.page_content for d in search_results])
                pages = sorted(list(set([d.metadata.get('page', 0) + 1 for d in search_results])))
                
                # B. Generation
                llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.1)
                full_prompt = f"Gunakan konteks ini: {context}\n\nPertanyaan: {prompt}"
                
                response = llm.invoke(full_prompt)
                answer = response.content
                
                # C. Final Format
                full_response = f"{answer}\n\n> 📍 **Referensi:** Halaman {', '.join(map(str, pages))}"
                st.markdown(full_response)
                
                # Simpan data untuk feedback
                st.session_state.last_query = prompt
                st.session_state.last_answer = full_response
                st.session_state.last_pages = pages
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                st.rerun()
    else:
        st.info("Upload PDF dulu.")

# Menampilkan Widget Feedback di bawah pesan terakhir
if "messages" in st.session_state and len(st.session_state.messages) > 0:
    if st.session_state.messages[-1]["role"] == "assistant":
        st.write("---")
        st.caption("Bantu kami meningkatkan akurasi: Apakah jawaban ini benar?")
        feedback = st.feedback("thumbs")
        
        if feedback is not None:
            save_feedback(
                user_query=st.session_state.last_query,
                ai_response=st.session_state.last_answer,
                rating=feedback,
                pages=st.session_state.last_pages
            )
            st.toast("Feedback tersimpan!", icon="✅")
