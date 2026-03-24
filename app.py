import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="AI PDF Chatbot (Groq Stable)",
    page_icon="📄",
    layout="wide"
)

# --- 2. PENGATURAN API KEY ---
# Pastikan Anda menggunakan GROQ_API_KEY di Streamlit Secrets
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
else:
    st.error("⚠️ GROQ_API_KEY tidak ditemukan! Atur di Settings > Secrets.")
    st.stop()

# --- 3. FUNGSI RAG (PROSES DOKUMEN) ---
@st.cache_resource # Agar model embedding tidak di-load berulang kali
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def process_pdf(uploaded_file):
    try:
        # Simpan file PDF sementara
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Load dan pecah PDF menjadi potongan teks
        loader = PyPDFLoader("temp.pdf")
        pages = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(pages)
        
        # Buat database vektor di dalam RAM
        embeddings = get_embeddings()
        vectorstore = Chroma.from_documents(
            documents=chunks, 
            embedding=embeddings,
            collection_name="pdf_chat_groq"
        )
        return vectorstore
    except Exception as e:
        st.error(f"Gagal memproses PDF: {e}")
        return None

# --- 4. ANTARMUKA CHAT (UI) ---
st.title("📄 Smart PDF Explorer")
st.markdown("Tanya jawab dengan dokumen Anda menggunakan **RAG + Groq Llama 3.1**")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar untuk area kontrol
with st.sidebar:
    st.header("📁 Unggah Dokumen")
    uploaded_pdf = st.file_uploader("Pilih file PDF", type="pdf")
    
    if st.button("🚀 Proses Dokumen"):
        if uploaded_pdf:
            with st.spinner("Sedang menganalisis PDF..."):
                st.session_state.vectorstore = process_pdf(uploaded_pdf)
                st.success("Dokumen siap dianalisis!")
        else:
            st.warning("Pilih file PDF terlebih dahulu.")
    
    if st.button("🗑️ Hapus Chat"):
        st.session_state.messages = []
        st.rerun()

# Menampilkan riwayat chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 5. LOGIKA TANYA JAWAB ---
if prompt := st.chat_input("Apa yang ingin Anda ketahui dari dokumen ini?"):
    # Tampilkan input user
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Respon AI
    if "vectorstore" in st.session_state:
        with st.chat_message("assistant"):
            with st.spinner("Berpikir cepat dengan Groq..."):
                try:
                    # 1. Ambil konteks paling relevan (Retrieval)
                    docs = st.session_state.vectorstore.similarity_search(prompt, k=4)
                    context = "\n\n".join([doc.page_content for doc in docs])
                    
                    # 2. Siapkan Model LLM (Llama 3.1 - Versi Terbaru)
                    llm = ChatGroq(
                        model_name="llama-3.1-8b-instant", 
                        temperature=0.1
                    )
                    
                    # 3. Gabungkan Konteks dengan Pertanyaan
                    full_prompt = f"""
                    Anda adalah asisten AI yang membantu. Jawablah pertanyaan user HANYA berdasarkan konteks dokumen di bawah ini.
                    Jika jawaban tidak ada dalam dokumen, katakan bahwa Anda tidak menemukannya.

                    KONTEKS DOKUMEN:
                    {context}

                    PERTANYAAN USER: 
                    {prompt}
                    """
                    
                    response = llm.invoke(full_prompt)
                    answer = response.content
                    
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                
                except Exception as e:
                    st.error(f"Terjadi kesalahan saat menghubungi Groq: {e}")
    else:
        st.info("💡 Silakan upload PDF dan klik 'Proses' di sidebar terlebih dahulu.")
