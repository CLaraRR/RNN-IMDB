from keras.datasets import imdb
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
import keras.preprocessing.text as TextProcess
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import SimpleRNN
import pickle

# set params
embedding_dim = 32
vocab_size = 2000
seq_length = 200
nb_epoch = 5
batchsize = 500

# load data
(x_train, y_train),(x_test,y_test) = imdb.load_data('imdb.npz', num_words=vocab_size) # imdb.load_data() download from website and store default in .keras/datasets/imdb.npz
                                                                                # if already exist, excute imdb.load_data('imdb.npz')
# print(x_train[0])
# print(type(x_train[0])) # output <class 'list'>
word_index = imdb.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
decodeID2word = lambda content: [reverse_word_index.get(i - 3, '?') for i in content]
sentences_decoded = [decodeID2word(content) for content in x_train]
# print(sentences_decoded[0])

# build vocabulary
token = Tokenizer(num_words=vocab_size) # a dict instance, the most frequent 2000 words
# token.fit_on_sequences(x_train) # build a dict from x_train
token.fit_on_texts(sentences_decoded)
with open('tokenizer.pickle', 'wb') as file:
    pickle.dump(token, file, protocol=pickle.HIGHEST_PROTOCOL)
x_train = sequence.pad_sequences(x_train, maxlen=seq_length, padding='post') # unify x_train list sequences into numpy.narrays of length 200
x_test = sequence.pad_sequences(x_test, maxlen=seq_length, padding='post') 
# print(x_train[0])
# print(type(x_train[0])) # output <class 'numpy.ndarray'>
# print(x_test[0])


# start build model
model = Sequential()

# embedding layer must be the first layer
model.add(Embedding(
    output_dim = embedding_dim, # whatever
    input_dim = vocab_size, # dict length
    input_length = seq_length, # sample length

))
model.add(Dropout(0.1))

# rnn layer
model.add(SimpleRNN(
    units=16,
))

# fully connected layer
model.add(Dense(
    units = 256, 
    activation = 'relu'
))
model.add(Dropout(0.1))
model.add(Dense(
    units = 1,
    activation = 'sigmoid' # get a result between 0-1, neg if >0.5, else pos
))

# compile model
model.compile(
    loss='binary_crossentropy',
    optimizer='Adam',
    metrics=['accuracy']
)

# train model
model.fit(
    x = x_train,
    y = y_train,
    epochs = nb_epoch,
    batch_size = batchsize,
    validation_data = (x_test,y_test)
)

model.save('model')


