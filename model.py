import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.sequence import pad_sequences
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

# Use only Tokenizer for consistent text processing
tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
tokenizer.fit_on_texts(doc)
sequences = tokenizer.texts_to_sequences(doc)

# Pad sequences to ensure consistent input length
max_length = 200
X = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.1, random_state=40)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.11, random_state=40)

vocab_size = len(tokenizer.word_index) + 1 # melihat ukuran vocabulary
embedding_vector_length = 128
model = Sequential()
early_stopping = EarlyStopping(
    monitor="val_accuracy",
    min_delta=0,
    patience=0,
    verbose=0,
    mode="max",
    baseline=0.93,
    restore_best_weights=True,
    start_from_epoch=0,
)

model.add(Embedding(vocab_size, embedding_vector_length, input_length=max_length))
model.add(Dropout(0.5))
model.add(Bidirectional(LSTM(128)))
model.add(Dense(128, activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))  # Changed to 3 units for 3 classes

# Build the model to show proper shapes
model.build(input_shape=(None, max_length))
model.summary()

opt = tf.keras.optimizers.Adam(learning_rate=0.001, weight_decay=1e-4)
model.compile(optimizer=opt, metrics=['accuracy'], loss='sparse_categorical_crossentropy')  # Changed loss function
history = model.fit(X_train, y_train, epochs=50, validation_data=(X_val, y_val), callbacks=[early_stopping])

model.save('model.keras')

loss, acc = model.evaluate(X_train, y_train, verbose=1)
print('Training Accuracy: %.3f' % acc)
print('Training Loss: %.3f' % loss)

print()

loss, acc = model.evaluate(X_val, y_val, verbose=1)
print('Validation Accuracy: %.3f' % acc)
print('Validation Loss: %.3f' % loss)

print()

loss, acc = model.evaluate(X_test, y_test, verbose=1)
print('Testing Accuracy: %.3f' % acc)
print('Testing Loss: %.3f' % loss)