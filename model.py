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


comment_df = pd.read_csv('sentiment_analysis.csv')
comments = comment_df["comment"]
labels = LabelEncoder().fit_transform(comment_df["sentiment"])

doc = []
stemmer = WordNetLemmatizer()
for comment in comments:
    
    # Lemmatization
    document = comment.split()
    document = [stemmer.lemmatize(word) for word in document]
    document = ' '.join(document)
    doc.append(document)

layer = tf.keras.layers.TextVectorization()
layer.adapt(doc)
vt = layer(doc).numpy()
# Split data
X_train, X_test, y_train, y_test = train_test_split(vt, labels, test_size=0.2, random_state=42) 
n_feature = X_train.shape[1]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(doc)
vocab_size = len(tokenizer.word_index) + 1 # melihat ukuran vocabulary
embedding_vector_length = 128
model = Sequential()
model.add(Embedding(vocab_size, embedding_vector_length, input_shape=(n_feature,)))
model.add(Dropout(0.2))
model.add(LSTM(128))
model.add(Dense(128, activation='tanh'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.summary()

opt = tf.keras.optimizers.Adam(learning_rate=0.1)
model.compile(optimizer=opt, metrics=['accuracy'], loss=tf.keras.losses.categorical_crossentropy)
history = model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))

loss, acc = model.evaluate(X_test, y_test, verbose=1)
print('Accuracy: %.3f' % acc)
print('Loss: %.3f' % loss)