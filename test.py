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
import os
from dotenv import load_dotenv
load_dotenv()
# Initialize HuggingFace embeddings
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

GROQ_API_KEY = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
HUGGINGFACEHUB_API_TOKEN = st.secrets.get("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not GROQ_API_KEY or not HUGGINGFACEHUB_API_TOKEN:
    st.error("Missing API keys! Please add them in .env or Streamlit Secrets.")
    st.stop()

# Initialize embeddings with Hugging Face API
embeddings = HuggingFaceEmbeddings(
    api_key=HUGGINGFACEHUB_API_TOKEN,
    model_name="all-MiniLM-L6-v2"
)


# Set up Streamlit
st.title("Conversational RAG With PDF Uploads and Chat History")
st.write("Upload PDFs and chat with their content")

# Initialize LLM
try:
    llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="deepseek-r1-distill-llama-70b")
except Exception as e:
    st.error(f"Failed to initialize LLM: {e}")
    st.stop()

# Chat interface
session_id = st.text_input("Session ID", value="default_session")

# Statefully manage chat history
if 'store' not in st.session_state:
    st.session_state.store = {}

# File uploader
#uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
predefined_pdfs = ["Health Montoring Box (CHATBOT).pdf"]  # actual file paths

# Process uploaded PDFs with caching
@st.cache_data
def load_and_process_pdfs(pdf_paths):
    documents = []
    for pdf_path in pdf_paths:
        try:
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            documents.extend(docs)
        except Exception as e:
            st.error(f"Error processing {pdf_path}: {e}")
    return documents


if predefined_pdfs:
    documents = load_and_process_pdfs(predefined_pdfs)
    st.success(f"Successfully processed {len(documents)} pages from {len(predefined_pdfs)} PDF(s).")

    # Split and create embeddings for the documents
    @st.cache_resource
    def generate_embeddings(_documents):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        splits = text_splitter.split_documents(_documents)
        vectorstore = FAISS.from_documents(splits, embeddings)
        return vectorstore

    try:
        vectorstore = generate_embeddings(documents)
        retriever = vectorstore.as_retriever()
    except Exception as e:
        st.error(f"Error generating embeddings: {e}")
        st.stop()

    # Contextualize question prompt
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    # Answer question prompt
    system_prompt = (
    "You are a medical assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question concisely. If the question is about medical readings "
    "and the values are abnormal or dangerous, clearly advise the user "
    "to consult a doctor immediately. "
    "If you don't know the answer, say that you don't know. "
    "Keep the answer to 1-2 sentences maximum."
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

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    def extract_final_answer(response: str) -> str:
         """Extracts the answer after </think> or returns the original response."""
         if "</think>" in response:
            return response.split("</think>")[-1].strip()
         return response.strip()

    # Get session history function
    def get_session_history(session: str) -> BaseChatMessageHistory:
        if session_id not in st.session_state.store:
            st.session_state.store[session_id] = ChatMessageHistory()
        return st.session_state.store[session_id]

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain, get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    # User input handling with a spinner
user_input = st.text_input("Your question:", key="user_input", on_change=lambda: st.session_state.update({"submitted": True}))

# Submit button
submit_pressed = st.button("Submit", key="submit_button")

if submit_pressed or st.session_state.get("submitted"):
    st.session_state["submitted"] = False  # Reset the flag after submission
    if user_input:
        with st.spinner("Generating response..."):
            try:
                session_history = get_session_history(session_id)
                response = conversational_rag_chain.invoke(
                    {"input": user_input},
                    config={"configurable": {"session_id": session_id}},
                )

                # Extract only the part after </think>
                full_answer = response['answer']
                if "</think>" in full_answer:
                    final_answer = full_answer.split("</think>")[-1].strip()
                else:
                    final_answer = full_answer.strip()

                # Display clean, bold answer
                st.markdown(f"**Answer:** {final_answer}")

                # Optionally show chat history
                with st.expander("View Chat History"):
                    st.write(session_history.messages)
            except Exception as e:
                st.error(f"Error generating response: {e}")


    # Clear chat history button
if st.button("Clear Chat History"):
    st.session_state.store[session_id] = ChatMessageHistory()
    st.session_state.store[session_id].messages = []  # Explicitly clearing messages
    st.success("Chat history cleared!")

#else:
    #st.warning("Please upload at least one PDF file to proceed.")
