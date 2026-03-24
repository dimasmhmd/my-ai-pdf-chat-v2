# 📄 Smart PDF Explorer: Advanced RAG with Groq & LangChain

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-v1.30+-ff4b4b.svg)
![LangChain](https://img.shields.io/badge/langchain-latest-green.svg)
![Groq](https://img.shields.io/badge/inference-Groq_Llama_3.1-orange.svg)

**Smart PDF Explorer** adalah aplikasi *Retrieval-Augmented Generation* (RAG) modern yang memungkinkan pengguna berinteraksi secara cerdas dengan dokumen PDF. Aplikasi ini mengombinasikan kecepatan inferensi **Groq Llama 3.1** dengan akurasi **HuggingFace Embeddings** untuk memberikan jawaban yang presisi disertai referensi nomor halaman.

---

## 🌟 Fitur Utama
- **⚡ Super Fast Inference**: Menggunakan LPU (Language Processing Unit) dari Groq untuk respon instan.
- **📍 Precise Citations**: AI tidak hanya menjawab, tapi juga memberitahu di halaman mana informasi tersebut ditemukan.
- **🧠 Optimized RAG**: Menggunakan teknik *Recursive Character Splitting* untuk menjaga konteks paragraf agar tidak terpotong.
- **🛡️ Robust Architecture**: Sistem tetap stabil meskipun terjadi pembatasan wilayah (geofencing) pada API tertentu dengan menggunakan *Local Embeddings*.

---

## 🛠️ Arsitektur Teknis
Sistem ini bekerja dengan alur kerja RAG (Retrieval-Augmented Generation) yang dioptimalkan:

1. **Ingestion**: Dokumen PDF dimuat dan dipecah menjadi chunk (~1200 karakter) dengan *overlap* untuk menjaga konteks.
2. **Embedding**: Menggunakan model `all-MiniLM-L6-v2` dari HuggingFace yang berjalan secara lokal di server.
3. **Vector Storage**: Koordinat teks disimpan dalam **ChromaDB** (in-memory) untuk pencarian cepat.
4. **Retrieval**: Mengambil top-K dokumen paling relevan menggunakan *Similarity Search*.
5. **Generation**: Mengirimkan konteks ke **Llama 3.1-8b** via Groq Cloud dengan *Prompt Engineering* yang ketat untuk mencegah halusinasi.

---

## 🚀 Panduan Instalasi

### 1. Prasyarat
- Python 3.11 atau lebih tinggi.
- API Key dari [Groq Cloud](https://console.groq.com/).

### 2. Kloning Repositori
```bash
git clone [https://github.com/username/smart-pdf-explorer.git](https://github.com/username/smart-pdf-explorer.git)
cd smart-pdf-explorer
