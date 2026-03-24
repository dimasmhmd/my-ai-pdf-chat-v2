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
st.set_page_config(page_title="Multi-Language AI PDF Pro", page_icon="🌐", layout="wide")

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
            documents=chunks, embedding=load_embeddings(), collection_name="multi_lang_rag"
        )
        return vectorstore
    except Exception as e:
        st.error(f"Error: {e}")
        return None

# ==========================================
# 4. ANTARMUKA PENGGUNA (UI)
# ==========================================
st.title("🌐 Multi-Language PDF Assistant")
st.caption("AI akan menjawab sesuai dengan bahasa yang Anda gunakan (Indonesia/Inggris).")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("📁 Upload Center")
    pdf_file = st.file_uploader("Select PDF file", type="pdf")
    if st.button("🚀 Proses"):
        if pdf_file:
            with st.spinner("Analyzing documents..."):
                st.session_state.vectorstore = process_pdf(pdf_file)
                st.success("Ready to be analyzed!")
    
    st.divider()
    if st.button("🗑️ Reset Chat"):
        st.session_state.messages = []
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ==========================================
# 5. LOGIKA TANYA JAWAB (MULTI-LANGUAGE)
# ==========================================
if prompt := st.chat_input("Tanyakan sesuatu..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if "vectorstore" in st.session_state:
        with st.chat_message("assistant"):
            with st.spinner("Searching..."):
                # A. Retrieval
                search_results = st.session_state.vectorstore.similarity_search(prompt, k=4)
                context = "\n\n".join([d.page_content for d in search_results])
                pages = sorted(list(set([d.metadata.get('page', 0) + 1 for d in search_results])))
                
                # B. Generation dengan Instruksi Bahasa Otomatis
                llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.1)
                
                # System Prompt Cerdas
                system_instruction = f"""
                You are a professional assistant. 
                1. Detect the language of the user's question.
                2. Respond ONLY in the same language as the question (e.g., if user asks in Indonesian, answer in Indonesian. If in English, answer in English).
                3. Base your answer strictly on the context provided below.
                4. If the answer is not in the context, politely state that you don't know in the user's language.

                CONTEXT:
                {context}
                """
                
                response = llm.invoke(system_instruction + f"\n\nQUESTION: {prompt}")
                answer = response.content
                
                # C. Tampilkan Jawaban & Source Preview (Opsi A: st.info)
                st.markdown(answer)
                
                with st.expander("🔍 Source Preview (Original Text)"):
                    for i, doc in enumerate(search_results):
                        p_num = doc.metadata.get('page', 0) + 1
                        st.markdown(f"**Source {i+1} - Page {p_num}:**")
                        st.info(f"\"{doc.page_content}\"")
                
                # D. Final Format untuk History & Logging
                source_ref = f"\n\n> 📍 Referensi: Halaman {', '.join(map(str, pages))}"
                full_res = answer + source_ref
                
                st.session_state.last_query = prompt
                st.session_state.last_answer = full_res
                st.session_state.last_pages = pages
                st.session_state.messages.append({"role": "assistant", "content": full_res})
                st.rerun()
    else:
        st.info("Silakan upload PDF terlebih dahulu.")

# Widget Feedback
if "messages" in st.session_state and len(st.session_state.messages) > 0:
    if st.session_state.messages[-1]["role"] == "assistant":
        st.write("---")
        feedback = st.feedback("thumbs")
        if feedback is not None:
            save_feedback(st.session_state.last_query, st.session_state.last_answer, feedback, st.session_state.last_pages)
            st.toast("Feedback saved!")
