from flask import Flask, render_template, request
import pickle
import nltk
from tokenizer import LemmaTokenizer  # Import from tokenizer module

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('wordnet')

# Load the saved model and vectorizer
model = pickle.load(open('modelLogReg.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

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
