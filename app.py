from flask import Flask, request, render_template
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import re
import pickle

app = Flask(__name__)

# Load model and vectorizer
with open("model.pkl", "rb") as f:
    model, vectorizer = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html', msg='')

@app.route('/predict', methods=['POST'])
def predict():
    sentence = request.form['tweet']
    nltk.download("stopwords")
    # Initialize the Porter Stemmer
    stemmer = PorterStemmer()

    stemmed_content = re.sub("[^a-zA-Z]", " ", sentence)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [stemmer.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = " ".join(stemmed_content)

    # Transform the content directly, no need to reshape
    stemmed_content_vectorized = vectorizer.transform([stemmed_content])
    # Prediction expects an array, not sparse matrix
    prediction = model.predict(stemmed_content_vectorized)[0]

    if prediction == 0:
        prediction_msg = "Tweet is Negative"
    else:
        prediction_msg = "Tweet is Positive"

    return render_template('msg.html', msg=prediction_msg)

if __name__ == '__main__':
    app.run(debug=True)
