from datetime import datetime
import pytz
import re
import dateparser
from langchain_core.tools import StructuredTool
from config import TIMEZONE
from booking.state import BookingState
from utils.csv_handler import save_booking_to_csv


booking_state = BookingState()

def get_current_datetime() -> str:
    now = datetime.now(pytz.UTC).astimezone(pytz.timezone(TIMEZONE))
    return f"🕰️ Current: {now.strftime('%Y-%m-%d %H:%M %Z')}"

def parse_date(date_str: str) -> str:
    parsed = dateparser.parse(
        date_str,
        settings={"PREFER_DATES_FROM": "future", "RELATIVE_BASE": datetime.now()}
    )
    return parsed.strftime("%Y-%m-%d") if parsed else None

def parse_time(time_str: str) -> str:
    time_str = time_str.lower().strip()
    if "noon" in time_str: return "12:00"
    if "midnight" in time_str: return "00:00"
    match = re.search(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", time_str, re.IGNORECASE)
    if not match: return None
    hour, minute, period = int(match.group(1)), int(match.group(2) or 0), match.group(3)
    if period:
        period = period.lower()
        if period == "pm" and hour != 12: hour += 12
        elif period == "am" and hour == 12: hour = 0
    return f"{hour:02d}:{minute:02d}"

def check_availability(date: str) -> str:
    iso_date = parse_date(date)
    if not iso_date:
        return "Could not understand the date. Try 'tomorrow' or 'sunday'."
    global booking_state
    booking_state.date = iso_date
    return f"✅ {iso_date} works. What time? (e.g., 3 PM, 12 noon)"

def check_time_availability(time: str) -> str:
    parsed_time = parse_time(time)
    if not parsed_time:
        return "Could not understand the time. Try '3 PM', '12 noon', or '15:00'."
    global booking_state
    booking_state.time = parsed_time
    return "Got it! What's your full name?"

def collect_name(name: str) -> str:
    if len(name.strip()) < 2:
        return "Please enter a valid full name."
    global booking_state
    booking_state.name = name.strip()
    return "Got it! What's your email address?"

def is_valid_email(email: str) -> bool:
    regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(regex, email) is not None

def collect_email(email: str) -> str:
    if not is_valid_email(email):
        return "Invalid email format. Example: name@example.com"
    global booking_state
    booking_state.email = email
    return "Thanks! What's your phone number? (e.g., +977-98XXXXXXX)"

def collect_phone(phone: str) -> str:
    global booking_state
    cleaned = re.sub(r"[^+\d]", "", phone)
    if not re.fullmatch(r"\+?[\d]{10,15}", cleaned):
        return "❌ Invalid phone. Use 10-15 digits (e.g., +1234567890)."
    booking_state.phone = cleaned
    try:
        save_booking_to_csv(booking_state)
        msg = (
            f"✅ Booked for {booking_state.name} on {booking_state.date} at {booking_state.time}!\n"
            f"📞 {booking_state.phone} | 📧 {booking_state.email}"
        )
    except Exception as e:
        msg = f"Failed to save booking: {str(e)}"
    # Reset
    booking_state = BookingState()
    return msg

# List of tools
tools = [
    StructuredTool.from_function(
        func=get_current_datetime, 
        name="GetCurrentDateTime", 
        description="Get current time"),
    
    StructuredTool.from_function(
        func=check_availability, 
        name="CheckAvailability", 
        description="Check date"),
    
    StructuredTool.from_function(
        func=check_time_availability, 
        name="CheckTimeAvailability", 
        description="Check time"),
    
    StructuredTool.from_function(
        func=collect_name, 
        name="CollectName", 
        description="Collect name"),
    
    StructuredTool.from_function(
        func=collect_email, name="CollectEmail", 
        description="Collect email"),
    
    StructuredTool.from_function(
        func=collect_phone, 
        name="CollectPhone", 
        description="Save appointment"),
]