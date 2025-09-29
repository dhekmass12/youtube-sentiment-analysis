import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
import nltk
import re
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer

nltk.download('wordnet')


comment_df = pd.read_csv('die_with_a_smile.csv')
comments = comment_df["comment"]
labels = LabelEncoder().fit_transform(comment_df["sentiment"])

doc = []
stemmer = WordNetLemmatizer()
for comment in comments:
    # Remove all the special characters
    document = re.sub(r'\W', ' ', str(comment))
   
    # Remove all single characters
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
    
    # Remove single characters from the start
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document) 
    
    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)
    
    # Removing prefixed 'b'
    document = re.sub(r'^b\s+', '', document)
    
    # Converting to Lowercase
    document = document.lower()
    
    # Lemmatization
    document = document.split()
    document = [stemmer.lemmatize(word) for word in document]
    document = ' '.join(document)
    doc.append(document)

layer = tf.keras.layers.TextVectorization()
layer.adapt(doc)
vt = layer(doc).numpy()
# Split data
X_train, X_test, y_train, y_test = train_test_split(vt, labels, test_size=0.3, random_state=42) 
n_feature = X_train.shape[1]