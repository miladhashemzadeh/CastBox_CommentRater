from keras.models import Sequential
from keras.layers import Conv1D, Dropout, Dense, Flatten, LSTM, MaxPooling1D, Bidirectional
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping, TensorBoard
import pandas
import random
import keras.backend as K
import numpy as np
import hazm
import fasttext

#should download model cc.fa.300.bin
model = fasttext.load_model("C:/Users/VampyreLord/PycharmProjects/CastBoxcomment_rate/Include/ML/res/cc.fa.300.bin")
csv_dataset = pandas.read_csv("C:/Users/VampyreLord/PycharmProjects/CastBoxcomment_rate/Include/ML/res/Tutorial_Dataset.csv")
revlist = list(map(lambda x: [x[0],x[1]],zip(csv_dataset['Text'],csv_dataset['Suggestion'])))
pos = list(filter(lambda x: x[1] == 1,revlist))
nat = list(filter(lambda x: x[1] == 2,revlist))
neg = list(filter(lambda x: x[1] == 3,revlist))
revlist_shuffle = pos[:450] + neg[:450]
random.shuffle(revlist_shuffle)
#@title Prepare Train & Test Data
vector_size = 300 #@param {type:"integer"}
max_no_tokens = 20 #@param {type:"integer"}

train_size = int(0.9*(len(revlist_shuffle)))
test_size = int(0.1*(len(revlist_shuffle)))

indexes = set(np.random.choice(len(revlist_shuffle), train_size + test_size, replace=False))

x_train = np.zeros((train_size, max_no_tokens, vector_size), dtype=K.floatx())
y_train = np.zeros((train_size, 2), dtype=np.int32)

x_test = np.zeros((test_size, max_no_tokens, vector_size), dtype=K.floatx())
y_test = np.zeros((test_size, 2), dtype=np.int32)

# @title Fill X_Train, X_Test, Y_Train, Y_Test with Dataset
print('title Fill X_Train, X_Test, Y_Train, Y_Test with Dataset')
for i, index in enumerate(indexes):
    text_words = hazm.word_tokenize(revlist_shuffle[index][0])
    for t in range(0, len(text_words)):
        if t >= max_no_tokens:
            break

        if text_words[t] not in model.words:
            continue
        if i < train_size:
            x_train[i, t, :] = model.get_word_vector(text_words[t])
        else:
            x_test[i - train_size, t, :] = model.get_word_vector(text_words[t])

    if i < train_size:
        y_train[i, :] = [1.0, 0.0] if revlist_shuffle[index][1] == 3 else [0.0, 1.0]
    else:
        y_test[i - train_size, :] = [1.0, 0.0] if revlist_shuffle[index][1] == 3 else [0.0, 1.0]

x_train.shape, x_test.shape, y_train.shape, y_test.shape

#@title Set batchSize and epochs
print('title Set batchSize and epochs')
batch_size = 500 #@param {type:"integer"}
no_epochs = 200 #@param {type:"integer"}
w2v_model = model
del model

model = Sequential()

model.add(Conv1D(32, kernel_size=3, activation='elu', padding='same',
                 input_shape=(max_no_tokens, vector_size)))
model.add(Conv1D(32, kernel_size=3, activation='elu', padding='same'))
model.add(Conv1D(32, kernel_size=3, activation='relu', padding='same'))
model.add(MaxPooling1D(pool_size=3))

model.add(Bidirectional(LSTM(512, dropout=0.2, recurrent_dropout=0.3)))

model.add(Dense(512, activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(512, activation='sigmoid'))
model.add(Dropout(0.25))
model.add(Dense(512, activation='sigmoid'))
model.add(Dropout(0.25))

model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001, decay=1e-6), metrics=['accuracy'])

# tensorboard = TensorBoard(log_dir='logs/', histogram_freq=0, write_graph=True, write_images=True)

model.summary()
model.fit(x_train, y_train, batch_size=batch_size, shuffle=True, epochs=no_epochs,
         validation_data=(x_test, y_test))
model.metrics_names
model.evaluate(x=x_test, y=y_test, batch_size=32, verbose=1)
model.save('persian-sentiment-fasttext.model')



if __name__ == "__main__":
    print("Posetive count {}".format(len(pos)))
    print("Negetive count {}".format(len(neg)))
    print("Natural  count {}".format(len(nat)))
    print()
    print("Total    count {}".format(len(revlist)))
    print()
    print("Posetive count : ", "\n", pos[random.randrange(1, len(pos))])
    print("Negetive count : ", "\n", neg[random.randrange(1, len(neg))])
    print("unknown  count : ", "\n", nat[random.randrange(1, len(nat))])
    print("Total    count {}".format(len(revlist_shuffle)))