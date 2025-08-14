import streamlit as st
from dotenv import load_dotenv
from rag import process_uploaded_pdfs
from booking.agent import create_booking_agent
from langchain_core.messages import HumanMessage, AIMessage
import sys
import os

# Load environment variables
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
if sys.platform.startswith("win"):
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


# Streamlit UI
st.title("📄 Simple Chatbot with Booking")

# Sidebar for PDF upload
with st.sidebar:
    st.header("📤 Upload PDF")
    uploaded_pdfs = st.file_uploader(
        "Upload PDF(s)", type="pdf", accept_multiple_files=True, key="pdf_uploader"
    )

    if uploaded_pdfs:
        st.success(f"{len(uploaded_pdfs)} file(s) uploaded!")
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
    st.session_state.rag_chain = None


# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# Handle user input
if prompt := st.chat_input("Ask or say 'book appointment'..."):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant response
    with st.chat_message("assistant"):
        lower_prompt = prompt.strip().lower()
        response = None

        # 1. Handle Greetings
        if any(greeting in lower_prompt for greeting in ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]):
            response = (
                "Hello! How can I assist you today?\n\n"
                "You can:\n"
                "- 📄 Ask questions about the **uploaded PDF**\n"
                "- 🗓️ Say _'book appointment'_ to schedule a call"
            )

        # 2. Handle Cancellation (only if in booking flow)
        elif st.session_state.get("in_booking", False) and any(
            cancel_word in lower_prompt for cancel_word in ["cancel", "never mind", "forget it", "stop", "exit", "no thanks"]
        ):
            st.session_state.in_booking = False
            response = "✅ Appointment booking cancelled. How else can I help?"

        # 3. Handle Booking Initiation
        elif any(word in lower_prompt for word in ["book", "appointment", "schedule", "reserve"]):
            st.session_state.in_booking = True
            try:
                agent_executor = create_booking_agent()
                result = agent_executor.invoke({
                    "input": prompt,
                    "chat_history": st.session_state.chat_history
                })
                response = result["output"]
                # If booking completes or fails, exit flow
                if "booked" in response.lower() or "❌" in response or "could not" in response.lower():
                    st.session_state.in_booking = False
            except Exception as e:
                response = f"Sorry, there was an error starting the booking: {str(e)}"
                st.session_state.in_booking = False

        # 4. Continue Booking Flow
        elif st.session_state.get("in_booking", False):
            try:
                agent_executor = create_booking_agent()
                result = agent_executor.invoke({
                    "input": prompt,
                    "chat_history": st.session_state.chat_history
                })
                response = result["output"]
                if "booked" in response.lower() or "❌" in response or "could not" in response.lower():
                    st.session_state.in_booking = False
            except Exception as e:
                response = f"Booking error: {str(e)}"
                st.session_state.in_booking = False

        # 5. Handle RAG Query (if PDF is uploaded)
        elif "rag_chain" in st.session_state and st.session_state.rag_chain is not None:
            try:
                result = st.session_state.rag_chain.invoke({
                    "question": prompt,
                    "chat_history": st.session_state.chat_history
                })
                answer = result.get("answer", "I don't know.")
                response = f"{answer}"
            except Exception as e:
                response = f"Error retrieving answer: {str(e)}"

        # 6. Default Fallback
        else:
            response = (
                "I can assist with:\n"
                "1. **Document Q&A**: Upload a PDF and ask questions about it\n"
                "2. **Booking appointments**: Say _'book appointment'_ to get started\n\n"
                "Try saying _'hello'_ or upload a PDF to begin!"
            )

        # Display and save assistant response
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.chat_history.extend([
            HumanMessage(content=prompt),
            AIMessage(content=response)
        ])


# Welcome message on first load
if not st.session_state.messages:
    welcome_msg = (
        "Hello! I'm your assistant.\n\n"
        "You can:\n"
        "- 📄 **Ask about documents**: Upload a PDF first\n"
        "- 🗓️ **Book an appointment**: Just say _'book appointment'_"
    )
    st.session_state.messages.append({"role": "assistant", "content": welcome_msg})
    st.rerun()
