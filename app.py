from flask import Flask, render_template,request, jsonify
import sys
# model library need for ml
import re
import pickle

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import json

# database 
import psycopg2
from psycopg2 import extras


# MySQL configuration
pg_config = {
    'host': '192.168.1.22',
    'database': 'bullymessage',
    'user': 'cyberman',
    'password': 'It123456',
    'port': 5432,
    'sslmode':"disable"

}

encodings_to_try = ['utf-8', 'latin-1', 'utf-8-sig']

app = Flask('__name__')

# load model from train data
model=pickle.load(open('detec_model_v1.pkl','rb'))
vectorizer=pickle.load(open('vectorizer_v1.pkl','rb'))

@app.route('/')
def start_web():
    return render_template("index.html")

@app.route('/training_process')
def training_process():
    return render_template("cyberbullying-detection.html")

@app.route('/detection',methods=['POST'])
def detection():
    if request.method == 'POST':
        # get data form input
        message = request.form.get("messageText")
        if(message!=""):
            # Clean data for predict
            text = clean_text(message)
            # return render back to index file
            return render_template("index.html", detect_result=text,input_message=message)
        
# Select messages from db
@app.route('/messages', methods=['GET'])
def get_messages():
    connection = None
    try:
        connection = get_db_connection()
        with connection.cursor(cursor_factory=extras.DictCursor) as cursor:
            cursor.execute("SELECT * FROM cybermessage ORDER BY record_date DESC LIMIT 100")
            results = cursor.fetchall()
            # Convert to list of dictionaries
            messages = [dict(row) for row in results]
        # return jsonify(messages), 200
        return render_template('index.html', 
                             messages=messages,
                             table_headers=['ID', 'Message', 'Type', 'Date'])
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if connection:
            connection.close()
@app.route('/messages_api', methods=['GET'])
def get_messages_api():
    connection = None
    try:
        connection = get_db_connection()
        with connection.cursor(cursor_factory=extras.DictCursor) as cursor:
            cursor.execute("SELECT * FROM cybermessage ORDER BY record_date DESC LIMIT 100")
            results = cursor.fetchall()
            # Convert to list of dictionaries
            messages = [dict(row) for row in results]
        return jsonify(messages), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if connection:
            connection.close()
def insert_message(message, message_type):
    connection = None
    try:
        # Validate required fields no need(*)
        if not message or not message_type:
            return jsonify({'error': 'Missing required fields'}), 400

        # Get database connection
        connection = get_db_connection()
        with connection.cursor(cursor_factory=extras.DictCursor) as cursor:
            query = """
                INSERT INTO cybermessage (message, message_type)
                VALUES (%s, %s)
                RETURNING id, message, message_type
            """
            cursor.execute(query, (message, message_type))
            results = cursor.fetchall()
            connection.commit()
            # Convert to list of dictionaries
            messages = [dict(row) for row in results]
            return messages, 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if connection:
            connection.close()
# Clean the data funtion
def clean_text(text):
    input_message = text
    # Remove HTML tags
    text = re.sub('<.*?>', '', str(text))

    # Remove non-alphabetic characters and convert to lowercase
    text = re.sub('[^a-zA-Z]', ' ', str(text)).lower()

    # Remove URLs, mentions, and hashtags from the text
    text = re.sub(r'http\S+', '', str(text))
    text = re.sub(r'@\S+', '', str(text))
    text = re.sub(r'#\S+', '', str(text))

    if(text == ''):
        return ''
    # Tokenize the text
    words = nltk.word_tokenize(text)

    all_stopwords = stopwords.words('english')
    # make stopwords keep key word

    # Remove stopwords
    words = [w for w in words if w not in all_stopwords]

    # Stem the words
    stemmer = PorterStemmer()
    words = [stemmer.stem(w) for w in words]

    # Join the words back into a string
    text = ' '.join(words)

    # New text Message from input
    new_message = [text]

    print(new_message)
    new_X_test = vectorizer.transform(new_message).toarray()
    # print(new_X_test)
    new_y_pred = model.predict(new_X_test)

    # return new_y_pred[0]
    # for real using uncomment message below
    if(new_y_pred[0]):
        insert_message(input_message,new_y_pred[0])
        return new_y_pred[0]

def get_db_connection():
    return psycopg2.connect(**pg_config)

# Create table if not exists (Run once)
with get_db_connection() as connection:
    with connection.cursor() as cursor:
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS cybermessage (
            id SERIAL PRIMARY KEY,
            message TEXT NOT NULL,
            message_type VARCHAR(50) NOT NULL,
            record_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        cursor.execute(create_table_sql)
        connection.commit()

# Clean the data funtion for new data dataset
def clean_text_newdata(text):
    input_message = text
    # Remove HTML tags
    text = re.sub('<.*?>', '', str(text))

    # Remove non-alphabetic characters and convert to lowercase
    text = re.sub('[^a-zA-Z]', ' ', str(text)).lower()

    # Remove URLs, mentions, and hashtags from the text
    text = re.sub(r'http\S+', '', str(text))
    text = re.sub(r'@\S+', '', str(text))
    text = re.sub(r'#\S+', '', str(text))

    if(text == ''):
        return ''
    # Tokenize the text
    words = nltk.word_tokenize(text)

    all_stopwords = stopwords.words('english')
    # make stopwords keep key word

    # Remove stopwords
    words = [w for w in words if w not in all_stopwords]

    # Stem the words
    stemmer = PorterStemmer()
    words = [stemmer.stem(w) for w in words]

    # Join the words back into a string
    text = ' '.join(words)

    # New text Message from input
    new_message = [text]

    print(new_message)
    new_X_test = vectorizer.transform(new_message).toarray()
    new_y_pred = model.predict(new_X_test)

    # return new_y_pred[0]
@app.route('/newdataset',methods=['GET'])
def setnewdata():
    connection = None
    for encoding in encodings_to_try:
        try:
            with open('aggression_parsed_dataset.json', 'r', encoding=encoding) as f:
                data = json.load(f)
            print("File loaded successfully with encoding:", encoding)
            break
        except UnicodeDecodeError:
            print(f"Failed with encoding: {encoding}")
        except json.JSONDecodeError as e:
            print("JSON format error:", e)
            break
    total = 0
    success = 0
    errors = 0
    for item in data:
            total += 1
            try:
                # Check key data before run to make sure it collect key binding value
                type = clean_text_newdata(item['Text'])
                if(type):
                    # Prepare values
                    values = (
                        item['Text'],
                        type,
                    )
                    # Validate required fields no need(*)
                    if not values:
                        return jsonify({'error': 'Missing required fields'}), 400

                    # Get database connection
                    connection = get_db_connection()
                    with connection.cursor(cursor_factory=extras.DictCursor) as cursor:
                        query = """
                            INSERT INTO cybermessage (message, message_type)
                            VALUES (%s, %s)
                            RETURNING id, message, message_type
                        """
                        cursor.execute(query, (values))
                        success += 1
            except (errors.DataError, errors.IntegrityError) as e:
                print(f"Database error with record {total}: {e}")
                errors += 1
                connection.rollback()
            except Exception as e:
                print(f"Error processing record {total}: {e}")
                errors += 1
            connection.commit()
            print(f"\nImport complete:")
            print(f"Total records: {total}")
            print(f"Successfully inserted: {success}")
            print(f"Failed: {errors}")


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)