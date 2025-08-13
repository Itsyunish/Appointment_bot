# config.py
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

# Load API Key

def get_api_key():
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(" GOOGLE_API_KEY not found in .env file")
    return api_key


def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.3,
        convert_system_message_to_human=True
    )

def get_embeddings():
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001")


# Constants

BOOKINGS_FILE = "bookings.csv"
TIMEZONE = "Asia/Kathmandu"