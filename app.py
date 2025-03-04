from flask import Flask, render_template,request
import sys
# model library need for ml
import re
import pickle
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder

app = Flask(__name__)

# load model from train data
model=pickle.load(open('detec_model.pkl','rb'))


@app.route('/')
def hello_world():
    return render_template("index.html")



@app.route('/detection',methods=['POST','GET'])
def detection():
    message = request.form.get("messageText")
   
    text = clean_text(message)

    detect = textToArray(text)
    print(detect)
    return render_template("index.html", detect_result=text)


# Clean the data
def clean_text(text):
    # Remove HTML tags
    text = re.sub('<.*?>', '', text)

    # Remove non-alphabetic characters and convert to lowercase
    text = re.sub('[^a-zA-Z]', ' ', text).lower()

    # Remove URLs, mentions, and hashtags from the text
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\S+', '', text)
    text = re.sub(r'#\S+', '', text)

    # Tokenize the text
    words = nltk.word_tokenize(text)

    # Remove stopwords
    words = [w for w in words if w not in stopwords.words('english')]

    # Stem the words
    stemmer = PorterStemmer()
    words = [stemmer.stem(w) for w in words]

    # Join the words back into a string
    text = ' '.join(words)
    print(text)
    return text



# Create the Bag of Words model
def textToArray(text):
    cv = CountVectorizer()
    X = cv.fit_transform(text)
    return X



if __name__ == "__main__":
    app.run(debug=True)