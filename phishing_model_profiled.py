import pandas as pd
import string
import re
import cProfile

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

def main():
    # Load dataset
    data = pd.read_csv("final_combined_dataset.csv")

    # Rename columns if needed
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

    # Train model
    model = MultinomialNB()
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print("✅ Model Evaluation:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Test on a sample email
    sample_email = """
    We have detected unusual activity on your account and need to verify your identity to ensure your account security.
    Please log in immediately by clicking the secure link below:
    http://secure-verify-update.com/login
    Failure to verify your account within 24 hours will result in permanent suspension.
    Thank you for your prompt attention to this matter.
    """
    result = predict_email(sample_email, model, vectorizer)
    if result == 1:
        print("🚨 Phishing Email detected!")
    else:
        print("✅ Legitimate Email.")

def predict_email(text, model, vectorizer):
    cleaned = clean_text(text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    return prediction

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()

    main()

    profiler.disable()
    profiler.dump_stats("callgraph.prof")
