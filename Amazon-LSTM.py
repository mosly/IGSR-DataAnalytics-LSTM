import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Load the dataset
current_folder = os.path.dirname(__file__)
file_path = os.path.join(current_folder, 'Amazon-Product.csv')
df = pd.read_csv(file_path, encoding='ISO-8859-1')

# Preprocess the data
X = df['review_body'].astype(str)
y = df['sentiment']

# Encode the target variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Tokenize the text data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)
X_sequences = tokenizer.texts_to_sequences(X)

# Pad sequences to ensure uniform length
max_sequence_length = max([len(seq) for seq in X_sequences])
X_pad = pad_sequences(X_sequences, maxlen=max_sequence_length)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_pad, y, test_size=0.2, random_state=42)

# Define the model
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=100))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_test, y_test))


# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
# New text data for prediction
new_texts = ["the battery goes dead more quickly than you'd expect", "Would definately recommend Fire HD"]
# Tokenize and pad the new text data
new_sequences = tokenizer.texts_to_sequences(new_texts)
new_padded_sequences = pad_sequences(new_sequences, maxlen=max_sequence_length)
# Make predictions
predictions = model.predict(new_padded_sequences)
# Convert predictions to sentiment labels
sentiment_labels = ['Positive' if pred > 0.5 else 'Negative' for pred in predictions]
# Print predictions
for text, label in zip(new_texts, sentiment_labels):
    print(f'Text: {text} --> Sentiment: {label}')

