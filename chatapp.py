import json
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder

import colorama
colorama.init()
from colorama import Fore, Style, Back

import random
import pickle

with open("intents.json") as file:
    data = json.load(file)


def chat():
    # load trained model
    model = keras.models.load_model('chat_model')

    # load tokenizer obeject
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    # load encoder object
    with open('label_encoder.pickle', 'rb') as en:
        labl_encoder = pickle.load(en)


    # parameters
    max_len = 20


    while True:
        print(Fore.LIGHTBLUE_EX + "User: " + Style.RESET_ALL, end="")
        inpt = input()
        if inpt.lower() == 'quit':
            break

        result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([inpt]),
                                             truncating='post', maxlen=max_len))

        tag = labl_encoder.inverse_transform([np.argmax(result)])


        for i in data['intents']:
            if i['tag'] == tag:
                print(Fore.GREEN + "ChatBot: " + Style.RESET_ALL, np.random.choice(i['responses']))

        # print(Fore.GREEN + "ChatBot: " + Style.RESET_ALL, np.random.choice('responses'))


print(Fore.YELLOW + "Welcome To My ChatBot Start Chat Here....!!!" + Style.RESET_ALL)
chat()
