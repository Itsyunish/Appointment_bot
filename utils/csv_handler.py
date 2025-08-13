# utils/csv_handler.py
import csv
import os

def save_booking_to_csv(booking, file_path="bookings.csv"):
    """Save booking details to CSV file"""
    file_exists = os.path.isfile(file_path)
    fieldnames = ["Date", "Time", "Name", "Email", "Phone"]

    with open(file_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "Date": booking.date,
            "Time": booking.time,
            "Name": booking.name,
            "Email": booking.email,
            "Phone": booking.phone
        })