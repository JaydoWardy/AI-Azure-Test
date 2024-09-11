import numpy as np
import pandas as pd
import random
import pickle
from nltk.tokenize import RegexpTokenizer #type: ignore
from tensorflow.keras.models import Sequential, load_model #type: ignore
from tensorflow.keras.layers import LSTM, Dense, Activation #type: ignore
from tensorflow.keras.optimizers import RMSprop #type: ignore
 
text_df = pd.read_csv("fake_or_real_news.csv")
text = list(text_df.text.values)
joined_text = " ".join(text)
 
partial_text = joined_text[:300000]
tokenizer = RegexpTokenizer(r"\w+")
tokens = tokenizer.tokenize(partial_text.lower())
 
unique_tokens = np.unique(tokens)
unique_token_index = {token: idx for idx, token in enumerate(unique_tokens)}
 
n_words = 10 #number of generated words change if u want :)
input_words = []
next_words = []
for i in range(len(tokens) - n_words):
    input_words.append(tokens[i:i + n_words])
    next_words.append(tokens[i + n_words])
 
X = np.zeros((len(input_words), n_words, len(unique_tokens)), dtype=bool) #Word then get the number of words, then assign 0 or 1 as to whether they have another word or not (bag of words type structure)
Y = np.zeros((len(next_words), len(unique_tokens)), dtype=bool)
 
for i, words in enumerate(input_words):
    for j, words in enumerate(words):
        X[i, j, unique_token_index[words]] = 1 #Getting the word from the specific index in the X position
    Y[i, unique_token_index[next_words[i]]] = 1 #Getting the singular word from the specific index in the Y position
 
model = Sequential()
model.add(LSTM(128, input_shape=(n_words, len(unique_tokens)), return_sequences=True))
model.add(LSTM(128))
model.add(Dense(len(unique_tokens)))
model.add(Activation("softmax"))
model.compile(loss='categorical_crossentropy', optimizer=RMSprop(learning_rate=0.01), metrics=['accuracy'])
model.fit(X, Y, batch_size=128, epochs=30, shuffle=True)
 
model.save("textgenmodel.keras")
 
model = load_model("textgenmodel.keras")
 
def predict_next_word(input_text, n_best):
    input_text = input_text.lower()
    X = np.zeros((1, n_words, len(unique_tokens)))
    for i, word in enumerate(input_text.split()):
        X[0, i, unique_token_index[word]] = 1
   
    predictions = model.predict(X)[0]
    return np.argpartition(predictions, -n_best) [-n_best:]
 
possible = predict_next_word("He will have to look into this thing and he", 5)
print([unique_tokens[idx] for idx in possible])
 
def generate_text (input_text, text_length, creativity=1):
    word_sequence = input_text.split()
    current = 0
    for _ in range(text_length):
        sub_sequence = " ".join(tokenizer.tokenize(" ".join(word_sequence).lower())[current:current+n_words])
        try:
            choice = unique_tokens[random.choice(predict_next_word(sub_sequence, creativity))]
        except:
            choice = random.choice(unique_tokens)
        word_sequence.append(choice)
        current += 1
    return " ".join(word_sequence)
generated_text = generate_text("He will have to look into this thing and he", 50, 1)
print(generated_text)