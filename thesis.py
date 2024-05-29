import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Sample data
data = pd.DataFrame({
    'comment': ["Great product, check this out!", "Buy now!", "I love this!", "Spam comment here."],
    'label': [0, 1, 0, 1]  # 0 for genuine, 1 for fake
})

# Preprocessing function
def preprocess(text):
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[^a-zA-Z]', ' ', text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    return text

data['clean_comment'] = data['comment'].apply(preprocess)

# Feature extraction
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['clean_comment'])
y = data['label']

# Model training
model = RandomForestClassifier()
model.fit(X, y)

# Model evaluation
y_pred = model.predict(X)
print(classification_report(y, y_pred))

# Detection function
def detect_fake_comment(comment):
    clean_comment = preprocess(comment)
    vectorized_comment = vectorizer.transform([clean_comment])
    prediction = model.predict(vectorized_comment)
    return prediction[0]

# Example usage
new_comment = "Amazing product! Buy now!"
result = detect_fake_comment(new_comment)
print("Fake" if result else "Genuine")
