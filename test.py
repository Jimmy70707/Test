import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
import warnings, os
from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=DeprecationWarning)
load_dotenv()

# API Key Loading
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
HUGGINGFACEHUB_API_TOKEN = st.secrets.get("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not GROQ_API_KEY or not HUGGINGFACEHUB_API_TOKEN:
    st.error("Missing API keys! Please add them in .env or Streamlit Secrets.")
    st.stop()

# Load model and embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Initialize LLM
try:
    llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="deepseek-r1-distill-llama-70b")
except Exception as e:
    st.error(f"Failed to initialize LLM: {e}")
    st.stop()

# Streamlit UI
st.title("📄 Conversational RAG with PDF Uploads + Chat History")
st.write("Upload medical PDFs and ask questions. Smart chat with context + memory.")

# Session ID
with st.sidebar:
    session_id = st.text_input("Session ID", value="default_session")
    if st.button("Clear Chat History"):
        st.session_state.store[session_id] = ChatMessageHistory()
        st.success("Chat history cleared!")

# Manage session chat memory
if 'store' not in st.session_state:
    st.session_state.store = {}

# PDF Setup
predefined_pdfs = ["Health Montoring Box (CHATBOT).pdf"]

@st.cache_data
def load_and_process_pdfs(pdf_paths):
    documents = []
    for pdf_path in pdf_paths:
        try:
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            documents.extend(docs)
        except Exception as e:
            st.error(f"Error loading {pdf_path}: {e}")
    return documents

@st.cache_resource
def generate_embeddings(_documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(_documents)
    return FAISS.from_documents(splits, embeddings)

# Load and embed PDFs
if predefined_pdfs:
    documents = load_and_process_pdfs(predefined_pdfs)
    st.success(f"Loaded {len(documents)} page(s) from PDF(s).")

    try:
        vectorstore = generate_embeddings(documents)
        retriever = vectorstore.as_retriever()
    except Exception as e:
        st.error(f"Embedding Error: {e}")
        st.stop()

    # Contextualizing user questions
    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given chat history and a follow-up question, rephrase it to be standalone."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_prompt)

    # Answering questions
    system_prompt = (
        "You are a helpful medical assistant. "
        "Use the context to answer clearly. "
        "If medical readings are dangerous, urge the user to consult a doctor. "
        "Say 'I don't know' if unsure.\n\n{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

    # Chat history manager
    def get_session_history(session: str) -> BaseChatMessageHistory:
        if session not in st.session_state.store:
            st.session_state.store[session] = ChatMessageHistory()
        return st.session_state.store[session]

    conversational_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    # Chat input
    user_input = st.text_input("Ask something about the document:", key="user_input")
    if st.button("Submit") and user_input:
        with st.spinner("Generating response..."):
            try:
                session_history = get_session_history(session_id)
                response = conversational_chain.invoke(
                    {"input": user_input},
                    config={"configurable": {"session_id": session_id}}
                )
                answer = response['answer'].split("</think>")[-1].strip()
                st.markdown(f"**Answer:** {answer}")
                with st.expander("View Chat History"):
                    st.write(session_history.messages)
            except Exception as e:
                st.error(f"Error: {e}")
