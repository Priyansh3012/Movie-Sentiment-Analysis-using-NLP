# Movie Review Sentiment Analysis Using NLP

## Description
This project focuses on sentiment analysis of movie reviews using **Natural Language Processing (NLP)** techniques. By leveraging advanced **text preprocessing, feature engineering, and machine learning models**, the application classifies reviews as either positive or negative. The project includes data visualization to understand the characteristics of the text data and to ensure an effective model-building process.
![image](https://github.com/user-attachments/assets/b0618b3e-c898-4e6f-bdbb-2669f5459309)
![image](https://github.com/user-attachments/assets/5cadca5d-2616-4248-bf44-68e392c990eb)

## Tech Stack
- **Libraries Used:**
  - **Data Analysis and Visualization:** `NumPy`, `pandas`, `seaborn`, `matplotlib`, `plotly`, `WordCloud`
  - **Text Processing and NLP:** `nltk`, `sklearn` (CountVectorizer, TfidfVectorizer), `re`, `string`, `unicodedata`
  - **Modeling and Evaluation:** `sklearn` (Logistic Regression, Decision Tree, Random Forest), `joblib`
  - **Frontend Development:** `Flask`, `HTML`, `CSS`

## Project Flow

1. **Data Cleaning and Preprocessing:**
   - **Removed special characters, URLs, and stopwords from text.**
   - **Expanded English contractions to standardize the text data.**

2. **Data Analysis and Visualization:**
   - **Checked for class imbalance using countplots**.
   - Analyzed and visualized the **most important words in positive and negative reviews using word clouds**.
   - **Compared review lengths, character counts, and word counts with histograms and density plots.**
   - Used **CountVectorizer and bag-of-words models to plot common unigrams, bigrams, trigrams, and four-grams for both positive and negative reviews.**

3. **Feature Engineering:**
   - **Applied word lemmatization** to standardize text data.
   - Trained models using different **n-grams (unigram, bigram, trigram, four-gram) with CountVectorizer and TfidfVectorizer**.
   - **Evaluated feature importance** and determined that the combination of unigrams, bigrams, and trigrams with TfidfVectorizer performed best.

4. **Model Building and Evaluation:**
   - Built and evaluated models using **Logistic Regression, Decision Tree, and Random Forest Classifiers**.
   - **Finalized Logistic Regression** as the best model due to its superior accuracy.
   - Evaluated model performance using classification metrics such as accuracy and **precision recall, accuracy, AUC score and F1-score**.

5. **Hyperparameter Tuning:**
   - Conducted **hyperparameter tuning for Logistic Regression using GridSearchCV.**
   - Identified **optimal parameters: `C=1.0`, `max_iter=100`, `penalty='l2'`, `tol=0.0001`**.
   - Retrained the model with these parameters and **achieved an accuracy of 84%** on validation data.

6. **Model Deployment:**
   - **Plotted the confusion matrix to visualize classification results.**
   - **Saved the finalized model using joblib.**
   - **Developed a user interface using Flask, HTML, and CSS**, allowing users to input movie reviews and receive sentiment predictions.

## Conclusion
The project successfully demonstrates the effectiveness of **NLP techniques** and **machine learning models** in analyzing movie review sentiments. By thoroughly preprocessing text data and employing rigorous model evaluation, the application provides reliable sentiment predictions, achieving an **84% accuracy rate** on validation data. The finalized Logistic Regression model, integrated with a user-friendly frontend, offers an intuitive way for users to analyze movie reviews in real time.
