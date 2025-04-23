import joblib
import re
import string
import tkinter as tk
from tkinter import messagebox

# Load trained model and vectorizer
model = joblib.load("phishing_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Clean and combine input fields
def clean_text(text):
    text = str(text).lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def predict_email():
    sender = sender_entry.get()
    receiver = receiver_entry.get()
    subject = subject_entry.get()
    body = body_entry.get("1.0", tk.END)
    urls = urls_entry.get()

    full_text = " ".join(
        clean_text(field) for field in [sender, receiver, subject, body, urls] if field
    )

    vectorized = vectorizer.transform([full_text])
    prediction = model.predict(vectorized)[0]

    if prediction == 1:
        messagebox.showwarning("Result", "🚨 This email is likely PHISHING.")
    else:
        messagebox.showinfo("Result", "✅ This email appears LEGITIMATE.")

# Build GUI
root = tk.Tk()
root.title("Test Your Email - Phishing Detector")
root.geometry("500x500")

# Layout
tk.Label(root, text="Sender").pack()
sender_entry = tk.Entry(root, width=50)
sender_entry.pack()

tk.Label(root, text="Receiver").pack()
receiver_entry = tk.Entry(root, width=50)
receiver_entry.pack()

tk.Label(root, text="Subject (optional)").pack()
subject_entry = tk.Entry(root, width=50)
subject_entry.pack()

tk.Label(root, text="Body").pack()
body_entry = tk.Text(root, width=60, height=10)
body_entry.pack()

tk.Label(root, text="URLs (optional)").pack()
urls_entry = tk.Entry(root, width=50)
urls_entry.pack()

tk.Button(root, text="Test Email", command=predict_email).pack(pady=10)

root.mainloop()
