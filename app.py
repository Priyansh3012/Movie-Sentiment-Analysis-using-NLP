from flask import Flask, render_template, request
import nltk
from joblib import dump, load
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
#from tokenizer import LemmaTokenizer  # Import from tokenizer module

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('wordnet')

# Defining LemmaTokenizer
def lemma_tokenizer(reviews):
    wordnetlemma = WordNetLemmatizer()
    return [wordnetlemma.lemmatize(word) for word in word_tokenize(reviews)]

# Loading the models
vectorizer = load('vectorizer.joblib')
model = load('modelLogReg.joblib')

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the review text from the form
        review = request.form['review']

        # Transform the review using the loaded vectorizer
        transformed_review = vectorizer.transform([review]).toarray()

        # Predict the sentiment using the loaded model
        prediction = model.predict(transformed_review)

        # Interpret the result
        sentiment = 'Positive' if prediction == '1' else 'Negative'

        return render_template('index.html', review=review, sentiment=sentiment)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
