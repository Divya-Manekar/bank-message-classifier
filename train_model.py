import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import pandas as pd

# Load your data (Ensure 'bank.csv' is in the same directory)
df = pd.read_csv('bank.csv')

# Clean up any leading/trailing spaces in column names
df.columns = df.columns.str.strip()

# Handle missing values in the 'text' column
df['text'] = df['text'].fillna('')

# Drop rows where the target column is missing
df = df.dropna(subset=['target'])

# Prepare features and labels
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(df['text']).toarray()

# Adjust this if your label column is different (make sure 'target' exists in your CSV)
y = df['target']  # Replace 'target' with your actual column name if different

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = MultinomialNB()
model.fit(X_train, y_train)

# Save the trained model and TF-IDF vectorizer to files
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

with open('tfidf.pkl', 'wb') as file:
    pickle.dump(tfidf, file)

print("Model and TF-IDF vectorizer trained and saved as 'model.pkl' and 'tfidf.pkl'")
