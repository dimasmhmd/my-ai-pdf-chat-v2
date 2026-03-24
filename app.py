import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="AI PDF Assistant Pro",
    page_icon="🤖",
    layout="wide"
)

# --- 2. PENGATURAN API KEY ---
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
else:
    st.error("⚠️ Masukkan GROQ_API_KEY di Streamlit Secrets!")
    st.stop()

# --- 3. FUNGSI RAG TEROPTIMASI ---
@st.cache_resource
def load_embedding_model():
    # Menggunakan model embedding lokal (HuggingFace) untuk stabilitas
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def process_pdf_optimized(uploaded_file):
    try:
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        loader = PyPDFLoader("temp.pdf")
        pages = loader.load()
        
        # OPTIMASI: Splitter dengan separators agar paragraf tidak terpotong kasar
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,      
            chunk_overlap=200,    
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = text_splitter.split_documents(pages)
        
        embeddings = load_embedding_model()
        
        # Simpan ke ChromaDB (In-Memory)
        vectorstore = Chroma.from_documents(
            documents=chunks, 
            embedding=embeddings,
            collection_name="pdf_optimized_db"
        )
        return vectorstore
    except Exception as e:
        st.error(f"Gagal memproses PDF: {e}")
        return None

# --- 4. ANTARMUKA PENGGUNA (UI) ---
st.title("📄 Smart PDF Assistant (Optimized)")
st.caption("Versi 2.0: Dilengkapi dengan Pelacakan Sumber Halaman & Akurasi Tinggi")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.header("⚙️ Kontrol Dokumen")
    uploaded_pdf = st.file_uploader("Unggah PDF (Maks 10MB)", type="pdf")
    
    if st.button("🚀 Mulai Analisis"):
        if uploaded_pdf:
            with st.spinner("Sedang memetakan isi dokumen..."):
                st.session_state.vectorstore = process_pdf_optimized(uploaded_pdf)
                st.success("Analisis selesai! AI sekarang memahami dokumen Anda.")
        else:
            st.warning("Pilih file PDF terlebih dahulu.")
    
    st.divider()
    if st.button("🗑️ Reset Percakapan"):
        st.session_state.messages = []
        st.rerun()

# Menampilkan Riwayat Chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 5. LOGIKA RETRIEVAL & GENERATION (OPTIMIZED) ---
if prompt := st.chat_input("Tanyakan sesuatu tentang isi dokumen..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if "vectorstore" in st.session_state:
        with st.chat_message("assistant"):
            with st.spinner("Mencari referensi terpercaya..."):
                # 1. RETRIEVAL: Mencari 5 potongan teks paling relevan
                search_results = st.session_state.vectorstore.similarity_search(prompt, k=5)
                
                # 2. CONTEXT BUILDING: Menggabungkan teks & mengambil metadata halaman
                context_parts = []
                pages_found = set()
                
                for doc in search_results:
                    context_parts.append(doc.page_content)
                    # Metadata halaman (tambah 1 karena index mulai dari 0)
                    page_num = doc.metadata.get("page", 0) + 1
                    pages_found.add(page_num)
                
                context_text = "\n\n---\n\n".join(context_parts)
                sorted_pages = sorted(list(pages_found))

                # 3. GENERATION: Memanggil Llama 3.1 melalui Groq
                try:
                    llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.1)
                    
                    system_instruction = f"""
                    Anda adalah asisten AI profesional. Jawablah pertanyaan user hanya berdasarkan konteks dokumen di bawah ini.
                    
                    KONTEKS DOKUMEN:
                    {context_text}
                    
                    INSTRUKSI:
                    1. Jika jawaban tidak ditemukan, katakan 'Maaf, saya tidak menemukan informasi tersebut di dokumen.'
                    2. Jawab dengan bahasa yang sopan dan terstruktur.
                    """
                    
                    response = llm.invoke(system_instruction + f"\n\nPERTANYAAN: {prompt}")
                    answer = response.content
                    
                    # Tambahkan footer Sumber Referensi Halaman
                    source_footer = f"\n\n> 📍 **Referensi:** Ditemukan pada halaman {', '.join(map(str, sorted_pages))}"
                    full_response = answer + source_footer
                    
                    st.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                
                except Exception as e:
                    st.error(f"Terjadi kendala pada otak AI: {e}")
    else:
        st.info("💡 Silakan unggah PDF dan klik 'Mulai Analisis' terlebih dahulu.")
