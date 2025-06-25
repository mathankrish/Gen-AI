import streamlit as st
import os
import concurrent.futures
import hashlib
import shutil
import tempfile
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- Streamlit Page Setup ---
st.set_page_config(page_title="RAG Chat Assistant", layout="wide")
st.markdown("""
    <style>
    .stChatMessage {
        padding: 0.75rem 1rem;
        border-radius: 1rem;
        margin-bottom: 1rem;
        display: inline-block;
        max-width: 80%;
    }
    .user-message {
        background-color: #DCF8C6;
        float: right;
        clear: both;
    }
    .bot-message {
        background-color: #F1F0F0;
        float: left;
        clear: both;
    }
    .chat-container {
        max-width: 800px;
        margin: auto;
        padding-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📚 RAG-Powered Chat Assistant")
st.caption("Upload your documents and start chatting with them like you're chatting with a person.")

# --- Helper Functions ---

def load_single_file(filepath):
    if filepath.endswith(".pdf"):
        return PyPDFLoader(filepath).load()
    elif filepath.endswith(".txt"):
        return TextLoader(filepath).load()
    elif filepath.endswith(".docx"):
        return Docx2txtLoader(filepath).load()
    return []

def load_documents_from_folder(directory_path):
    filepaths = [os.path.join(directory_path, f) for f in os.listdir(directory_path)
                 if f.endswith((".pdf", ".txt", ".docx"))]
    documents = []
    progress = st.progress(0, text="Loading 0%")
    total = len(filepaths)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for i, result in enumerate(executor.map(load_single_file, filepaths)):
            documents.extend(result)
            percent = int(((i + 1) / total) * 100)
            progress.progress((i + 1) / total, text=f"Loading {percent}%")
    return documents

def get_directory_hash(directory_path):
    hasher = hashlib.sha256()
    for root, _, files in os.walk(directory_path):
        for file in sorted(files):
            if file.endswith((".pdf", ".txt", ".docx")):
                filepath = os.path.join(root, file)
                hasher.update(file.encode())
                hasher.update(str(os.path.getmtime(filepath)).encode())
    return hasher.hexdigest()

def get_text_chunks(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(documents)

def get_vector_store(text_chunks, index_path="faiss_index"):
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    progress = st.progress(0, text="Indexing 0%")
    chunk_batches = [text_chunks[i:i + 100] for i in range(0, len(text_chunks), 100)]
    all_vectors = []
    for i, batch in enumerate(chunk_batches):
        vectors = FAISS.from_documents(batch, embedding=embeddings)
        all_vectors.append(vectors)
        percent = int(((i + 1) / len(chunk_batches)) * 100)
        progress.progress((i + 1) / len(chunk_batches), text=f"Indexing {percent}%")

    final_store = all_vectors[0]
    for vs in all_vectors[1:]:
        final_store.merge_from(vs)

    final_store.save_local(index_path)
    return final_store

def get_chat_chain(vector_store):
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer')
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7, openai_api_key=OPENAI_API_KEY)
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory,
        return_source_documents=True
    )

def clear_cache(index_path="faiss_index", hash_path=".hash"):
    if Path(index_path).exists():
        shutil.rmtree(index_path)
    if Path(hash_path).exists():
        os.remove(hash_path)

# --- Session State ---
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "doc_names" not in st.session_state:
    st.session_state.doc_names = []
if "all_documents" not in st.session_state:
    st.session_state.all_documents = []

# --- Sidebar Document Loader ---
with st.sidebar:
    st.header("📂 Load Documents")
    data_dir = st.text_input("Document folder path", value="docs")
    uploaded_files = st.file_uploader("Upload individual documents", accept_multiple_files=True, type=["pdf", "txt", "docx"])

    selected_doc = st.selectbox("Choose a document for focused questions (optional):", ["All Documents"] + st.session_state.doc_names)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear Cache"):
            clear_cache()
            st.session_state.all_documents = []
            st.success("Cache cleared!")

    with col2:
        if st.button("Process Documents"):
            new_documents = []
            if os.path.isdir(data_dir):
                current_hash = get_directory_hash(data_dir)
                if Path(".hash").exists():
                    with open(".hash", "r") as f:
                        old_hash = f.read()
                else:
                    old_hash = ""

                if current_hash != old_hash:
                    with st.spinner("Updating vector store with folder documents..."):
                        new_documents_from_folder = load_documents_from_folder(data_dir)
                        new_documents.extend(new_documents_from_folder)
                        with open(".hash", "w") as f:
                            f.write(current_hash)

            if uploaded_files:
                with tempfile.TemporaryDirectory() as tmpdir:
                    for file in uploaded_files:
                        filepath = os.path.join(tmpdir, file.name)
                        with open(filepath, "wb") as f:
                            f.write(file.read())
                        new_documents_from_upload = load_single_file(filepath)
                        new_documents.extend(new_documents_from_upload)

            if new_documents:
                # Ensure new files are stacked (preserved)
                st.session_state.all_documents.extend(new_documents)
                chunks = get_text_chunks(st.session_state.all_documents)
                st.session_state.vector_store = get_vector_store(chunks)
                st.session_state.conversation = get_chat_chain(st.session_state.vector_store)
                st.session_state.doc_names = list(set([doc.metadata.get("source", "unknown") for doc in st.session_state.all_documents]))
                st.success("Vector store updated with all documents!")
            else:
                st.info("No valid documents found to process.")

# --- Chat Interface ---
if st.session_state.conversation:
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

    for message in st.session_state.chat_history:
        role = message["role"]
        css_class = "user-message" if role == "user" else "bot-message"
        with st.container():
            st.markdown(f"<div class='stChatMessage {css_class}'>{message['content']}</div>", unsafe_allow_html=True)

    user_input = st.chat_input("Type your message here...")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.spinner("Thinking..."):
            if selected_doc != "All Documents":
                user_input += f" Only refer to the document named {selected_doc}."

            response = st.session_state.conversation({"question": user_input})
            answer = response["answer"]
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)
else:
    st.info("Please load and process your documents first from the sidebar.")
