import streamlit as st
from dotenv import load_dotenv
from rag import process_uploaded_pdfs
from booking.agent import create_booking_agent
from langchain_core.messages import HumanMessage, AIMessage
import sys
import os


# get api
load_dotenv()
from config import get_api_key

try:
    api_key = get_api_key()
    os.environ["GOOGLE_API_KEY"] = api_key
except ValueError:
    st.warning("Set `GOOGLE_API_KEY` in `.env` file.")
    api_key = st.text_input("Enter your Google API Key:", type="password")
    if not api_key:
        st.stop()
    os.environ["GOOGLE_API_KEY"] = api_key



# Async fix for Windows
import sys
import asyncio
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())



# streamlit UI
st.set_page_config(page_title="🤖 Simple Chatbot", layout="wide")
st.title("📄 Simple Chatbot with Booking")

# Sidebar
with st.sidebar:
    st.header("📤 Upload PDF")
    uploaded_pdfs = st.file_uploader(
        "Upload PDF(s)", type="pdf", accept_multiple_files=True, key="pdf_uploader"
    )

    if uploaded_pdfs:
        st.success(f"✅ {len(uploaded_pdfs)} file(s) uploaded!")
        st.cache_resource.clear()
        with st.spinner("Processing PDFs..."):
            rag_chain = process_uploaded_pdfs(uploaded_pdfs)
            if rag_chain:
                st.session_state.rag_chain = rag_chain
                st.success("RAG ready!")
            else:
                st.error("Failed to process PDFs.")
    else:
        st.info("Upload a PDF for document Q&A")

   

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.chat_history = []
    st.session_state.in_booking = False

# Display chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle input
if prompt := st.chat_input("Ask or say 'book appointment'..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        lower_prompt = prompt.strip().lower()

        # Booking flow
        if st.session_state.get("in_booking", False):
            agent_executor = create_booking_agent()
            try:
                result = agent_executor.invoke({
                    "input": prompt,
                    "chat_history": st.session_state.chat_history
                })
                response = result["output"]
                if "booked" in response.lower() or "❌" in response:
                    st.session_state.in_booking = False
            except Exception as e:
                response = f"Booking error: {str(e)}"
                st.session_state.in_booking = False

        elif any(word in lower_prompt for word in ["book", "appointment", "schedule", "reserve"]):
            st.session_state.in_booking = True
            agent_executor = create_booking_agent()
            try:
                result = agent_executor.invoke({
                    "input": prompt,
                    "chat_history": st.session_state.chat_history
                })
                response = result["output"]
            except Exception as e:
                response = f"Booking error: {str(e)}"
                st.session_state.in_booking = False

        # RAG
        elif "rag_chain" in st.session_state:
            try:
                result = st.session_state.rag_chain.invoke({
                    "question": prompt,
                    "chat_history": st.session_state.chat_history
                })
                answer = result.get("answer", "I don't know.")
                response = f"{answer}\n\n📌 Tip: You can book an appointment by saying _'book appointment'_."
            except Exception as e:
                response = f"Error retrieving answer: {str(e)}"

        else:
            response = (
                "I can only assist with:\n"
                "1. 📄 Questions about the **uploaded PDF**\n"
                "2. 🗓️ **Booking appointments**\n\n"
                "Please upload a PDF or say _'book appointment'_ to proceed."
            )

        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.chat_history.extend([
            HumanMessage(content=prompt),
            AIMessage(content=response)
        ])

# Welcome message
if not st.session_state.messages:
    welcome_msg = (
        "Hello! I can help with:\n\n"
        "- 📄 **Document Q&A**: Upload a PDF and ask questions\n"
        "- 🗓️ **Booking**: Say _'book appointment'_ to schedule a call"
    )
    st.session_state.messages.append({"role": "assistant", "content": welcome_msg})
    st.rerun()