from flask import Flask, render_template, request
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import load
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('wordnet')

def lemma_tokenizer(reviews):
    wordnetlemma = WordNetLemmatizer()
    return [wordnetlemma.lemmatize(word) for word in word_tokenize(reviews)]

# Define the vectorizer
tfidfvect = TfidfVectorizer(
    analyzer="word",
    tokenizer=lemma_tokenizer,  # Use the function here
    ngram_range=(1, 3),
    min_df=10,
    max_features=5000
)

# Load the logistic regression model
model = load('modelLogReg.joblib')

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    try:
        if request.method == 'POST':
            review = request.form['review']

            # Check if the review is empty or None
            if not review:
                logging.error("No review provided.")
                return render_template('index.html', error="Please enter a review.")

            # Assuming the vectorizer has been fitted previously
            # transformed_review = tfidfvect.transform([review]).toarray()

            # For demo purposes, fitting on a placeholder list if needed
            # This should be done with your actual training data ideally
            placeholder_data = ["This is a good movie.", "I did not like this film."] # Example data
            tfidfvect.fit(placeholder_data)

            transformed_review = tfidfvect.transform([review]).toarray()
            logging.debug(f"Transformed Review Shape: {transformed_review.shape}")

            prediction = model.predict(transformed_review)
            sentiment = 'Positive' if prediction[0] == 1 else 'Negative'

            return render_template('index.html', review=review, sentiment=sentiment)

    except Exception as e:
        logging.error(f"Error during processing: {e}")
        return render_template('index.html', error="An error occurred during processing.")

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
