import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(page_title="Resume Chat", page_icon="📄", layout="wide")
st.title("📄 Chat with Resume")
st.markdown("Ask questions about Nitin Dumka's experience, skills, and projects.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Groq API Key", type="password")
    chunk_size = st.slider("Chunk Size", 300, 1000, 800)
    k = st.slider("K Similarity", 2, 10, 5)

    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

@st.cache_resource
def load_pdf(path, size):
    loader = PyPDFLoader(path)
    splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=150)
    chunks = splitter.split_documents(loader.load())
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(chunks, embedding=embeddings)
    return vectorstore, len(chunks)

pdf_path = "Nitin_Dumka_AI_ML_Engineer_Resume.pdf"

if api_key:
    vectorstore, count = load_pdf(pdf_path, chunk_size)
    st.success(f"Loaded {count} chunks!")

    llm = ChatGroq(model="llama-3.1-8b-instant", api_key=api_key, temperature=0.3)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer only from the resume. Chat history:\n{history}"),
        ("human", "Context: {context}\n\nQuestion: {question}")
    ])

    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(chat["question"])
        with st.chat_message("assistant"):
            st.write(chat["answer"])

    question = st.chat_input("Ask a question...")

    if question:
        with st.chat_message("user"):
            st.write(question)

        with st.spinner("Thinking..."):
            docs = vectorstore.similarity_search(question, k=k)
            context = "\n\n".join([d.page_content for d in docs])

            history_text = ""
            for chat in st.session_state.chat_history[-4:]:
                history_text += f"User: {chat['question']}\nBot: {chat['answer']}\n"

            response = llm.invoke(prompt.format_messages(
                history=history_text,
                context=context,
                question=question
            ))

        with st.chat_message("assistant"):
            st.write(response.content)

        st.session_state.chat_history.append({
            "question": question,
            "answer": response.content
        })

        with st.expander("Sources"):
            for i, doc in enumerate(docs):
                st.text(doc.page_content[:300])
else:
    st.info("Enter your Groq API key in the sidebar to start.")