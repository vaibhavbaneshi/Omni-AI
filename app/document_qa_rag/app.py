## RAG Q&A Conversation With PDF Including Chat History

import tempfile
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from common.streamlit_imports import st
from common.langchain_imports import PyMuPDFLoader, RecursiveCharacterTextSplitter, create_history_aware_retriever, create_retrieval_chain, create_stuff_documents_chain, HuggingFaceEmbeddings, ChatGroq, Chroma, ChatPromptTemplate, MessagesPlaceholder, RunnableWithMessageHistory, BaseChatMessageHistory, ChatMessageHistory
from common.standard_lib_imports import os, io
from utils.chunk_size import get_dynamic_chunk_size

os.environ['HF_TOKEN']=os.getenv("HF_TOKEN")
groq_api_key=os.getenv("GROQ_API_KEY")
embeddings=HuggingFaceEmbeddings(model_name="Sybghat/all-MiniLM-L6-v2-finetuned-squad")

st.title('RAG Q&A Conversation With PDF Including Chat History')
st.write('This app allows you to ask questions about a PDF document and maintain a chat history.')

llm=ChatGroq(groq_api_key=groq_api_key, model_name='Gemma2-9b-It')
session_id=st.text_input("Session ID", value="default_session")

if 'store' not in session_id:
    st.session_state.store={}

uploaded_files=st.file_uploader("Choose a PDF file", type='pdf', accept_multiple_files=True)

if uploaded_files and len(uploaded_files) > 0:
    all_docs = []

    for uploaded_file in uploaded_files:
        try:
            st.write(f"🔄 Processing: {uploaded_file.name}")
            file_bytes = uploaded_file.read()

            # Write to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file_bytes)
                temp_path = tmp.name

            # Load using file path
            loader = PyMuPDFLoader(temp_path)
            docs = loader.load()
            all_docs.extend(docs)

            st.success(f"✅ Loaded {len(docs)} pages from {uploaded_file.name}")

        except Exception as e:
            st.error(f"❌ Error loading {uploaded_file.name}: {e}")

    if all_docs:
        chunk_size = get_dynamic_chunk_size(all_docs)
        st.write(f"🧩 Chunk size: {chunk_size}")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size)
        splits = text_splitter.split_documents(all_docs)
        st.success(f"✅ Total chunks created: {len(splits)}")
    else:
        st.warning("⚠️ No documents were successfully loaded.")

    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory="./chroma_db")
    retriever = vectorstore.as_retriever()    

    contextualize_q_system_prompt=(
        "Given a chat history and the latest user question"
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    contextualize_q_prompt=ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever=create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    question_answer_chain=create_stuff_documents_chain(llm, qa_prompt)

    rag_chain=create_retrieval_chain(history_aware_retriever,question_answer_chain)

    def get_session_history(session:str)->BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id]=ChatMessageHistory()
            return st.session_state.store[session_id]
        
    conversational_rag_chain=RunnableWithMessageHistory(
        rag_chain,get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    user_input = st.text_input("Your question:")
    if user_input:
        session_history=get_session_history(session_id)
        response = conversational_rag_chain.invoke(
            {"input": user_input},
            config={
                "configurable": {"session_id":session_id}
            },  # constructs a key "abc123" in `store`.
        )
        st.write("Assistant:", response['answer'])
else:
    st.info("⬆️ Please upload one or more PDF files.")

