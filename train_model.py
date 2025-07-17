import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

# Load dataset directly from Kaggle
df = pd.read_csv("IMDB Dataset.csv")

# Map sentiments to binary
df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})
X = df['review']
y = df['label']

# Vectorize text
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_vec = vectorizer.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.3, random_state=42)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save model and vectorizer
joblib.dump(model, "sentiment_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

# Print evaluation report
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))