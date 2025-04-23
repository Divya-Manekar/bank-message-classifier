from flask import Flask, request, jsonify
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load and prepare data
df = pd.read_csv('bank.csv')

# Fill missing text values to avoid NaN errors
df['text'] = df['text'].fillna('')

# Initialize and fit the TF-IDF vectorizer
tfidf = TfidfVectorizer()
tfidf.fit(df['text'])

# Load your trained model (make sure model.pkl exists)
model = pickle.load(open('model.pkl', 'rb'))



# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return '''
        <h2>Text Prediction Form</h2>
        <form action="/predict" method="post">
            <textarea name="text" rows="4" cols="50" placeholder="Enter your text here"></textarea><br><br>
            <input type="submit" value="Predict">
        </form>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form.get('text')
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    vector_input = tfidf.transform([text]).toarray()
    result = model.predict(vector_input)[0]
    return jsonify({'target': str(result)})

if __name__ == '__main__':
    app.run(debug=True)