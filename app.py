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
st.set_page_config(page_title="AI PDF Translator Pro", page_icon="🌍", layout="wide")

if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
else:
    st.error("⚠️ GROQ_API_KEY tidak ditemukan di Secrets!")
    st.stop()

# ==========================================
# 2. FUNGSI LOGGING (LLMOps)
# ==========================================
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
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
        chunks = text_splitter.split_documents(pages)
        vectorstore = Chroma.from_documents(
            documents=chunks, embedding=load_embeddings(), collection_name="translator_rag"
        )
        return vectorstore
    except Exception as e:
        st.error(f"Error: {e}")
        return None

# ==========================================
# 4. ANTARMUKA PENGGUNA (UI)
# ==========================================
st.title("🌍 AI PDF Multi-Language & Translator")
st.markdown("Baca, Tanya, dan Terjemahkan dokumen PDF ke berbagai bahasa secara instan.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- SIDEBAR SETTINGS ---
with st.sidebar:
    st.header("📁 Upload & Settings")
    pdf_file = st.file_uploader("Pilih file PDF", type="pdf")
    
    if st.button("🚀 Proses Dokumen"):
        if pdf_file:
            with st.spinner("Menganalisis dokumen..."):
                st.session_state.vectorstore = process_pdf(pdf_file)
                st.success("Dokumen siap!")
        else:
            st.warning("Upload PDF terlebih dahulu.")

    st.divider()
    
    # Fitur Translate
    st.header("🌐 Fitur Terjemahan")
    enable_translate = st.checkbox("Aktifkan Paksa Terjemahan")
    target_lang = st.selectbox(
        "Pilih Bahasa Tujuan:",
        ["Indonesia", "English", "Japanese", "Mandarin", "Arabic", "German", "French", "Korean"],
        disabled=not enable_translate
    )

    st.divider()
    if st.button("🗑️ Hapus Riwayat"):
        st.session_state.messages = []
        st.rerun()

# Menampilkan Riwayat Chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ==========================================
# 5. LOGIKA TANYA JAWAB & TRANSLATE
# ==========================================
if prompt := st.chat_input("Tanyakan sesuatu atau minta terjemahkan bagian tertentu..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if "vectorstore" in st.session_state:
        with st.chat_message("assistant"):
            with st.spinner("Sedang memproses..."):
                # A. Retrieval
                search_results = st.session_state.vectorstore.similarity_search(prompt, k=4)
                context = "\n\n".join([d.page_content for d in search_results])
                pages = sorted(list(set([d.metadata.get('page', 0) + 1 for d in search_results])))
                
                # B. Generation dengan Logic Bahasa
                llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.1)
                
                # Menentukan Instruksi Bahasa
                if enable_translate:
                    lang_instruction = f"ALWAYS respond and translate the answer into {target_lang}."
                else:
                    lang_instruction = "Respond in the same language used by the user in their question."

                system_prompt = f"""
                You are a professional PDF assistant and translator.
                
                INSTRUCTIONS:
                1. {lang_instruction}
                2. Base your answer strictly on the provided CONTEXT.
                3. If the user asks to translate a specific part, find it in the context and translate it accurately.
                4. If the information is missing, state it clearly in the target language.

                CONTEXT:
                {context}
                """
                
                response = llm.invoke(system_prompt + f"\n\nUSER QUESTION: {prompt}")
                answer = response.content
                
                # C. Tampilkan Jawaban
                st.markdown(answer)
                
                # D. Preview Sumber (st.info)
                with st.expander("🔍 Lihat Referensi Asli (Source Preview)"):
                    for i, doc in enumerate(search_results):
                        p_num = doc.metadata.get('page', 0) + 1
                        st.markdown(f"**Sumber {i+1} - Halaman {p_num}:**")
                        st.info(f"\"{doc.page_content}\"")
                
                # E. Simpan ke History
                source_ref = f"\n\n> 📍 Referensi: Halaman {', '.join(map(str, pages))}"
                full_res = answer + source_ref
                
                st.session_state.last_query = prompt
                st.session_state.last_answer = full_res
                st.session_state.last_pages = pages
                st.session_state.messages.append({"role": "assistant", "content": full_res})
                st.rerun()
    else:
        st.info("💡 Silakan upload PDF dan klik 'Proses Dokumen' terlebih dahulu.")

# Widget Feedback
if "messages" in st.session_state and len(st.session_state.messages) > 0:
    if st.session_state.messages[-1]["role"] == "assistant":
        st.write("---")
        feedback = st.feedback("thumbs")
        if feedback is not None:
            save_feedback(st.session_state.last_query, st.session_state.last_answer, feedback, st.session_state.last_pages)
            st.toast("Feedback disimpan!")
