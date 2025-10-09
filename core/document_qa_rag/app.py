import os, re, fitz, tempfile, hashlib, asyncio
import streamlit as st
from concurrent.futures import ThreadPoolExecutor

from common.langchain_imports import PyMuPDFLoader, RecursiveCharacterTextSplitter, Chroma
from langchain.prompts import PromptTemplate
from langchain.retrievers import BM25Retriever, EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from utils.chunk_size import get_dynamic_chunk_size
from utils.llm import llm
from utils.embeddings import embeddings

# ========== SAFE INVOKE WITH SUMMARIZATION FALLBACK ==========
def safe_llm_invoke(question, context, qa_prompt, llm):
    try:
        return llm.invoke(qa_prompt.format(context=context, question=question))
    except Exception as e:
        err_str = str(e).lower()
        if any(term in err_str for term in ["too large", "413", "rate_limit", "too many tokens"]):
            st.warning("⚠️ Context too large — summarizing and retrying...")
            summarize_prompt = PromptTemplate.from_template(
                "Summarize the following context into key points. "
                "Preserve important names, numbers, and facts.\n\n{context}"
            )
            try:
                summary = llm.invoke(summarize_prompt.format(context=context[:10000]))
                st.info("🧾 Context summarized successfully.")
                return llm.invoke(qa_prompt.format(context=summary, question=question))
            except Exception as inner_e:
                return f"❌ Retry failed: {inner_e}"
        return f"❌ LLM error: {e}"

# ========== MAIN RAG FUNCTION ==========
def run_pdf_rag():
    st.title("PDF Q&A Assistant")

    uploaded_files = st.file_uploader("📄 Upload PDF files", type="pdf", accept_multiple_files=True)
    if not uploaded_files:
        st.info("⬆️ Upload PDF(s) to start.")
        return

    all_docs, metadata_store = [], {}

    with st.spinner("🔍 Extracting PDFs and metadata..."):
        for uf in uploaded_files:
            bytes_data = uf.read()
            file_hash = hashlib.md5(bytes_data).hexdigest()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(bytes_data)
                tmp_path = tmp.name

            with fitz.open(tmp_path) as pdf:
                meta = pdf.metadata
                metadata_store[uf.name] = {"meta": meta, "pages": pdf.page_count}

            loader = PyMuPDFLoader(tmp_path)
            docs = loader.load()
            for i, d in enumerate(docs):
                d.metadata["source"] = uf.name
                d.metadata["page"] = i + 1
                d.metadata["priority"] = 1.2 if i < 3 else 1.0
            all_docs.extend(docs)

    # Parallel cleanup
    def clean_doc(d):
        text = d.page_content.strip()
        if not text or re.search(r"copyright by", text, re.I) or len(text.split()) < 5:
            return None
        return d

    with ThreadPoolExecutor() as executor:
        clean_docs = list(filter(None, executor.map(clean_doc, all_docs)))

    chunk_size = get_dynamic_chunk_size(clean_docs)
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=200)
    splits = splitter.split_documents(clean_docs)

    # Cache per-file embeddings
    cache_dir = f"./chroma_cache/{file_hash}"
    if os.path.exists(cache_dir):
        st.info(f"♻️ Using cached vectorstore for {uf.name}")
        vectorstore = Chroma(persist_directory=cache_dir, embedding_function=embeddings)
    else:
        os.makedirs(cache_dir, exist_ok=True)
        vectorstore = Chroma.from_documents(splits, embeddings, persist_directory=cache_dir)

    semantic_retriever = vectorstore.as_retriever(search_kwargs={"k": 12})
    bm25_retriever = BM25Retriever.from_documents(splits)
    bm25_retriever.k = 12

    multi_prompt = PromptTemplate.from_template(
        "Generate 2 rephrasings of this question using different words but same meaning.\n\n{question}"
    )
    multi_retriever = MultiQueryRetriever.from_llm(retriever=semantic_retriever, llm=llm, prompt=multi_prompt)
    compressor = LLMChainExtractor.from_llm(llm)
    hybrid_retriever = EnsembleRetriever(
        retrievers=[
            ContextualCompressionRetriever(base_compressor=compressor, base_retriever=multi_retriever),
            bm25_retriever
        ],
        weights=[0.7, 0.3]
    )

    qa_prompt = PromptTemplate.from_template(
        "You are a document assistant. Use the provided context (including metadata) to answer accurately. "
        "If not found, reply 'Not mentioned'.\n\nContext:\n{context}\n\nQuestion: {question}"
    )

    st.markdown("### 💬 Ask Questions About Your PDFs")
    user_input = st.text_area("Enter your question:")
    if not st.button("🚀 Query PDF") or not user_input.strip():
        return

    with st.spinner("🤔 Thinking..."):
        docs = hybrid_retriever.get_relevant_documents(user_input)
        metadata_context = "\n".join([
            f"📘 {f}: {d['meta'].get('title','N/A')} | Author: {d['meta'].get('author','N/A')} | Pages: {d['pages']}"
            for f, d in metadata_store.items()
        ])
        text_context = "\n\n---\n\n".join([d.page_content for d in docs])
        full_context = metadata_context + "\n\n---\n\n" + text_context

        # Token-aware context limiter
        if len(full_context) > 12000:
            st.warning("⚠️ Context too large, trimming to safe limit.")
            full_context = full_context[:12000]

        response = safe_llm_invoke(user_input, full_context, qa_prompt, llm)
        st.success(response.content)