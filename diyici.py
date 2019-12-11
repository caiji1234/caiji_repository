import pickle
import numpy as np
import pandas as pd
# import keras_applications
# import keras as ke
# from keras_applications import vgg16
from keras.utils import np_utils
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.layers import LSTM, Dense, Embedding, Dropout



def load_data(filepath, input_shape):
    df = pd.read_csv(filepath)
    labels, vocabulary = list(df['label'].unique()), list(df['sentence'].unique())
    string = ''
    for word in vocabulary:
        string += word

    vocabulary = set(string)
    word_dictionary = {word: i+1 for i, word in enumerate(vocabulary)}

    with open('word_dic.pk', 'wb')as f:
        pickle.dump(word_dictionary, f)
    inverse_word_dictionary = {i+1: word for i, word in enumerate(vocabulary)}
    label_dictionary = {label:  i for i, label in enumerate(labels)}
    with open('label_dic.pk', 'wb')as f:
        pickle.dump(label_dictionary, f)
    output_dictionary = {i: label for i, label in enumerate(labels)}

    vocab_size = len(word_dictionary.keys())  # 词汇表大小
    label_size = len(label_dictionary.keys())  # 标签大小

    x = [[word_dictionary[word] for word in sent] for sent in df['sentence']]
    x = pad_sequences(maxlen=input_shape, sequences=x, padding='post', value=0)
    y = [[label_dictionary[sent]]for sent in df['label']]
    y = [np_utils.to_categorical(label, num_classes=label_size) for label in y]
    y = np.array([list(_[0]) for _ in y])

    return x, y, output_dictionary, vocab_size, label_size, inverse_word_dictionary


def creat_LSTM(n_units, input_shape, output_dim, filepath):
    x, y, output_dictionary, vocab_size, label_size, inverse_word_dictionary = load_data(filepath, input_shape)
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size+1, output_dim=output_dim, input_length=input_shape, mask_zero=True))
    model.add(LSTM(n_units, input_shape=(x.shape[0], x.shape[1])))
    model.add(Dropout(0.2))
    model.add(Dense(label_size, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    return model


def model_train(input_shape, filepath,  model_save_path):
    x, y, output_dictionary, vocab_size, label_size, inverse_word_dictionary = load_data(filepath, input_shape)
    n_units = 100
    batch_size = 32
    epochs = 5
    output_dim = 20

    # 模型训练
    lstm_model = creat_LSTM(n_units, input_shape, output_dim, filepath)
    lstm_model.fit(x, y, epochs=epochs, batch_size=batch_size, verbose=1)

    lstm_model.save(model_save_path)


filepath = 'C:\\Users\\cainiao\\Desktop\\train.csv'
input_shape = 50
model_save_path = './sentiment_analysis.h5'
model_train(input_shape, filepath, model_save_path)