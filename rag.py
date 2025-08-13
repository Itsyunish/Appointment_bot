# rag.py
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
import streamlit as st
from config import get_llm, get_embeddings
import os

@st.cache_resource(show_spinner=False)
def process_uploaded_pdfs(uploaded_files):
    try:
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as temp_dir:
            docs = []
            for file in uploaded_files:
                temp_path = os.path.join(temp_dir, file.name)
                with open(temp_path, "wb") as f:
                    f.write(file.getbuffer())
                loader = PyPDFLoader(temp_path)
                docs.extend(loader.load())

            if not docs:
                return None

            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.split_documents(docs)
            vectorstore = FAISS.from_documents(chunks, embedding=get_embeddings())

            return ConversationalRetrievalChain.from_llm(
                llm=get_llm(),
                retriever=vectorstore.as_retriever(search_kwargs={"k": 4})
            )
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        return None