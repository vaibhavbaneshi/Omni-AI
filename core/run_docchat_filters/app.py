import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from common.streamlit_imports import st

from common.langchain_imports import RecursiveCharacterTextSplitter
from common.langchain_imports import HuggingFaceEmbeddings
from common.langchain_imports import Chroma
from common.langchain_imports import PromptTemplate

from utils.llm import llm
from utils.file_loader_manager import FileLoaderManager

import pandas as pd

# LLM RAG wrapper with metadata included
def run_llm(query, docs):
    try:
        context_parts = []
        for d in docs:
            meta = f"[Source: {d.metadata.get('source')} | Type: {d.metadata.get('file_type')} | Page: {d.metadata.get('page', 'N/A')}]"
            context_parts.append(f"{meta}\n{d.page_content}")

        context = "\n\n".join(context_parts)

        template = """
                You are a professional AI research assistant. 
                Your task is to answer the user’s question using ONLY the information from the provided documents. 
                Follow these rules carefully:

                1. Base your answer strictly on the given context. Do not use outside knowledge.  
                2. If the documents do not contain enough information, politely explain that the answer is not available in the uploaded material and suggest what type of document or detail might help.  
                3. Always cite your sources clearly using their filename (and page number if available).  
                4. Provide concise, accurate, and well-structured responses.  
                5. If multiple documents provide relevant information, synthesize them into a single coherent answer.  

                ---
                📌 Question: {question}

                📚 Context (from documents with metadata):
                {context}

                ✍️ Answer (with references):
                """
        
        prompt = PromptTemplate.from_template(template)
        formatted = prompt.format(question=query, context=context)

        response = llm.invoke(formatted)
        return response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        st.error(f"LLM invocation failed: {e}")
        return "Error: Could not generate answer."

# Main Streamlit DocChat function
def run_docchat_with_filters():
    st.header("📄 DocChat with Metadata Filters (Multi-Format)")

    try:
        uploaded_files = st.file_uploader(
            "Upload one or more documents (PDF, DOCX, TXT, CSV, XLSX, PPTX, HTML, JSON, EPUB, Code, etc.)",
            type=["pdf", "docx", "txt", "md", "csv", "xlsx", "xls", "pptx", "html", "htm", "json",
                  "epub", "py", "js", "java", "cpp", "sql", "png", "jpg", "jpeg", "tiff"],
            accept_multiple_files=True
        )
    except Exception as e:
        st.error(f"File uploader failed: {e}")
        return

    if uploaded_files:
        loader = FileLoaderManager(uploaded_by="User")
        all_docs = []

        for f in uploaded_files:
            try:
                with open(f.name, "wb") as temp_file:
                    temp_file.write(f.read())
                docs = loader.load(f.name)
                all_docs.extend(docs)
            except Exception as e:
                st.error(f"Failed to process {f.name}: {e}")

        if not all_docs:
            st.warning("⚠️ No valid documents were loaded.")
            return

        try:
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.split_documents(all_docs)
        except Exception as e:
            st.error(f"Document chunking failed: {e}")
            return

        try:
            embeddings = HuggingFaceEmbeddings(
                model_name="BAAI/bge-small-en-v1.5",
                model_kwargs={"device": "cpu"}
            )
            persist_dir = "./chroma_db"
            vectordb = Chroma.from_documents(chunks, embeddings, persist_directory=persist_dir)
            vectordb.persist()
        except Exception as e:
            st.error(f"Vectorstore or embeddings initialization failed: {e}")
            return

        # ✅ Success message after upload + indexing
        st.success(f"✅ {len(uploaded_files)} document(s) uploaded and processed successfully!")

        # Metadata filter
        try:
            file_options = sorted(set([d.metadata["source"] for d in chunks]))
            selected_file = st.selectbox("📂 Filter by file", options=["All"] + file_options)
        except Exception as e:
            st.error(f"Failed to generate file filter options: {e}")
            selected_file = "All"

        filters = {}
        if selected_file != "All":
            filters["source"] = selected_file

        # ✅ Query section with a button
        st.subheader("🔎 Ask a Question")
        query = st.text_input("Enter your question:")

        if st.button("Ask Query"):
            if query.strip() == "":
                st.warning("⚠️ Please enter a question before submitting.")
            else:
                try:
                    retriever_kwargs = {"k": 5}
                    if filters:
                        retriever_kwargs["filter"] = filters

                    retriever = vectordb.as_retriever(search_kwargs=retriever_kwargs)
                    results = retriever.get_relevant_documents(query)

                    if not results:
                        st.warning("No relevant documents found for your query.")
                        return
                except Exception as e:
                    st.error(f"Document retrieval failed: {e}")
                    return

                # Deduplicate sources
                unique_sources = {}
                for r in results:
                    src = r.metadata["source"]
                    if src not in unique_sources:
                        unique_sources[src] = r

                # Sources table
                table_data = []
                for r in unique_sources.values():
                    snippet = r.page_content[:200].replace("\n", " ") + "..."
                    table_data.append([
                        r.metadata["source"],
                        r.metadata.get("file_type", "N/A"),
                        r.metadata.get("page", "N/A"),
                        r.metadata.get("timestamp", "N/A")[:19],
                        snippet
                    ])
                st.subheader("📄 Sources")
                st.table(pd.DataFrame(table_data, columns=["Source", "Type", "Page", "Uploaded", "Snippet"]))

                # LLM answer
                response_text = run_llm(query, results)
                st.subheader("🤖 Answer")
                st.markdown(f"{response_text}\n\n*Source(s): {', '.join(unique_sources.keys())}*")