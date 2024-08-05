from flask import Flask, render_template, request
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import load
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('wordnet')

# Load the logistic regression model
model = load('modelLogReg.joblib')

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        review = request.form['review']
        
        # Make sure to fit the vectorizer before transforming if not serialized
        # Example if data is available:
        # tfidfvect.fit(data)  # Use your training data

        transformed_review = tfidfvect.transform([review]).toarray()
        prediction = model.predict(transformed_review)
        sentiment = 'Positive' if prediction == 1 else 'Negative'

        return render_template('index.html', review=review, sentiment=sentiment)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
