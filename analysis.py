import csv
import pickle
import numpy as np
import pandas as pd
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences


with open('word_dict.pk', 'rb') as f:
    word_dictionary = pickle.load(f)
with open('Label_dict.pk', 'rb') as f:
    output_dictionary = pickle.load(f)

df = pd.read_csv('C:\\Users\\cainiao\\Desktop\\test.csv')
data = []
num = []
lstm_model = load_model('./sentiment_analysis.h5')
for sent in df['sentence']:
    try:
        input_shape = 50
        x = [[word_dictionary[word] for word in sent]]
        x = pad_sequences(maxlen=input_shape, sequences=x, padding='post', value=0)
        y_predict = lstm_model.predict(x)
        label_dict = {v: k for k, v in output_dictionary.items()}
        label_predict = label_dict[np.argmax(y_predict)]
        data.append(label_dict[np.argmax(y_predict)])
        num.append(sent)
        # print(label_dict[np.argmax(y_predict)])
    except KeyError as err:
        a = '没有相应符号'
        data.append(a)
        num.append(sent)


c = list(zip(num, data))
print(c)


with open('submission.csv', 'w',errors='ignore') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerow(('sentence', 'label'))
    writer.writerows((c))




