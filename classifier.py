import pandas as pd
import string
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

# Clean email body text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)  # remove URLs
    text = re.sub(r"\S+@\S+", "", text)  # remove emails
    text = re.sub(r"\d+", "", text)      # remove digits
    text = text.translate(str.maketrans("", "", string.punctuation))  # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()  # normalize whitespace
    return text

# Load dataset
data = pd.read_csv("final_combined_dataset.csv")

# Rename target column for model clarity
data = data.rename(columns={"body": "text", "label": "label"})

# Preprocess text
data["text"] = data["text"].astype(str).apply(clean_text)

# Split into features and labels
X = data["text"]
y = data["label"]

# Vectorize text
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.95)
X_vec = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# Train Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print("✅ Model Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Predict new email
def predict_email(text):
    cleaned = clean_text(text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    return prediction

# Example test email
if __name__ == "__main__":
    sample_email = """
    Dear user, Your account has been compromised. Click here immediately to reset your password and avoid suspension.
    """
    result = predict_email(sample_email)
    if result == 1:
        print("🚨 Phishing Email detected!")
    else:
        print("✅ Legitimate Email.")
