from flask import Flask, render_template,request
import sys
# model library need for ml
import re
import pickle
import joblib
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer



from sklearn.metrics import accuracy_score
# train a Logistic Regression Model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()

app = Flask(__name__)

# load model from train data
model=pickle.load(open('detec_model_v1.pkl','rb'))
vectorizer=pickle.load(open('vectorizer_v1.pkl','rb'))

@app.route('/')
def hello_world():
    return render_template("index.html")



@app.route('/detection',methods=['POST'])
def detection():
    if request.method == 'POST':
        message = request.form.get("messageText")

        text = clean_text(message)
        # print(text)
        # detect = textToArray(text)

        # print(detect)
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

    new_message = [text]

    print(new_message)
    new_X_test = vectorizer.transform(new_message).toarray()
    print(new_X_test)
    new_y_pred = model.predict(new_X_test)
    # print(new_y_pred)
    return new_y_pred



# Create the Bag of Words model
# def textToArray(text):
#     cv = CountVectorizer()
#
#     new_message = [text]
#     new_X_test = cv.fit(new_message).toarray()
#
#     new_y_pred = model.predict(new_X_test.reshape(-1,1))
#     print(new_y_pred)
#     return new_y_pred
# 1. Logistic Regression Model
# def detectClass(text_array):
#     clf = LogisticRegression(max_iter = 1000)
#     clf.fit(X_train, y_train)

#     prediction = model.predict(text_array)
#     print(prediction)
#     return prediction

if __name__ == "__main__":
    app.run(debug=True)