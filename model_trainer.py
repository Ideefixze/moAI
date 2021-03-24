import keras.utils.np_utils as np_utils
import tensorflow as tf
from keras.callbacks import Callback
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras import Input
from keras.layers import BatchNormalization as BatchNorm
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras.layers import Embedding

import numpy as np
import pandas as pd

import glob 
import random
import io

import json
import argparse

seq_len = 16
separate_symbol = '#'
prediction_limit = 2000

def get_model(input_shape, output_len) -> Sequential:

    model = Sequential()
    model.add(Embedding(output_len, output_len, input_length=seq_len))
    model.add(LSTM(512, input_shape=(input_shape[1], input_shape[2]),return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(512))
    model.add(Dropout(0.2))
    model.add(Dense(output_len, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])

    return model

def get_corpus(filename, count:int=2500) -> str:
    df = pd.read_csv(filename, index_col=None, header=0)

    def eliminator(item):
        try:
            return len(item)
        except:
            return 0

    #mix it up
    df = df.sample(frac=1)
    df = df[df['content'].map(eliminator) >= 2]
    if count<=0:
        corpus_text = separate_symbol.join(df['content'].tolist())
    else:
        corpus_text = separate_symbol.join(df['content'].tolist()[0:count])
    return corpus_text


def get_symbol_dict(corpus:str, save=True) -> dict:
    #get all set of used symbols
    symbols = set()
    for char in corpus:
        symbols.add(char)
    #create dictionary for symbols
    symbols = sorted(symbols)
    symbol_to_int = dict((p, id) for id, p in enumerate(symbols))

    #save for later use for ai_cmds
    if save:
        a_file = open("meta.json", "w+")
        json.dump(symbol_to_int, a_file)
        a_file.close()

    return symbol_to_int

def translate(sequence, symbol_dict:dict, int_to_letters=False) -> list:
    s_dict = symbol_dict
    if int_to_letters==True:
        s_dict = dict({v: k for k, v in symbol_dict.items()})

    translated = []
    for c in sequence:
        try:
            translated.append(s_dict[c])
        except Exception as e:
            print(f"Error while translating: \'{c}\' is not in the symbol dict.")
            print(s_dict)
            raise e
            
    return translated

def create_seqs_predictions(sequence:list):
    net_in = list()
    net_out = list()
    for i in range(0, len(sequence)-seq_len):
        s_in = sequence[i:i+seq_len]
        s_out = sequence[i+seq_len]
        net_in.append([symbol for symbol in s_in])
        net_out.append(s_out)

    return net_in, net_out

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def predict(model, s_dict, start_pattern, max_p_len = 400, temperature=1.0) -> str:
        while len(start_pattern) < seq_len:
            start_pattern = " " + start_pattern
        
        pattern = translate(start_pattern,s_dict)
        prediction_output = []

        max_p_len = min(prediction_limit, max_p_len)
        
        for note_index in range(max_p_len):
            prediction_input = np.reshape(pattern, (1, len(pattern), 1))
            #prediction_input = prediction_input / float(len(s_dict)) 

            prediction = model.predict(prediction_input, verbose=0)[0]    

            index = sample(prediction,temperature)
            result = index
            if s_dict[separate_symbol]==index: #if we find separate symbol, dont predict any further
                break

            prediction_output.append(result)    

            pattern = np.append(pattern, index)
            pattern = pattern[1:len(pattern)]
        
        out = translate(prediction_output,s_dict,True)
        out = ''.join(out)

        return out

class PeriodicPredict(Callback):

    def __init__(self,symbol_dict, inputs, p_len=100, temperature=0.3, period=5):
        super().__init__()
        self.symbol_dict = symbol_dict
        self.inputs = inputs
        self.period = period
        self.p_len = p_len
        self.temperature = temperature
    
    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.period != 0:
            return

        print("Performing epoch prediction...")
        
        pattern = np.concatenate(self.inputs[random.randint(0,len(self.inputs)-1)]).tolist()
        pattern = translate(pattern,self.symbol_dict,int_to_letters=True)
        out = predict(self.model,self.symbol_dict, pattern, self.p_len,self.temperature)

        print(out)
        f = open(f"outs/prediction_{epoch}.txt", "w+")
        f.write(out)
        f.close()


if __name__=="__main__":
    #parse arguments
    parser = argparse.ArgumentParser(description='Train the model.')
    parser.add_argument('--input_filename', type=str, required=True,
                        help='Filename of *.csv data file from \'data\' folder.')
    parser.add_argument('--data_lines_count', type=int, default=2500,
                        help='Number of random messages to be trained on.')
    parser.add_argument('--model', type=str, required=True,
                        help='Model name. If already you have model, it will continue training.')
    parser.add_argument('--epochs', type=int, default=75,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size.')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='Save every X epochs.')
    parser.add_argument('--pred_freq', type=int, default=10,
                        help='Do sample prediction every X epochs.')
    parser.add_argument('--pred_len', type=int, default=100,
                        help='Sample prediction length.')
    parser.add_argument('--pred_temp', type=float, default=0.3,
                        help='Sample prediction temperature.')

    args = parser.parse_args()

    #init data
    corpus = get_corpus(f"data/{args.input_filename}",count=args.data_lines_count)
    s_dict = get_symbol_dict(get_corpus(f"data/{args.input_filename}",count=-1))

    print(f"Corpus length: {len(corpus)}    Dict size: {len(s_dict)}")
    translated_corpus = translate(corpus,s_dict)

    net_in,net_out = create_seqs_predictions(translated_corpus)

    net_in = np.reshape(net_in, (len(net_in),seq_len,1))

    #normalization, uncomment if needed
    #net_in = net_in / float(len(s_dict))

    net_out = np_utils.to_categorical(net_out)
    
    #get out base model
    model = get_model(net_in.shape,len(s_dict))

    print(net_in.shape)
    print(model.input_shape)

    filepath = f"models/{args.model}.hdf5"

    checkpoint = ModelCheckpoint(
    filepath, monitor='loss', 
    verbose=0,        
    save_best_only=True,        
    mode='min',
    period=args.save_freq
    )

    callbacks_list = [checkpoint, PeriodicPredict(s_dict, net_in,p_len=args.pred_len, temperature=args.pred_temp, period=args.pred_freq)]   

    try:
        model.load_weights(f"models/{args.model}.hdf5")
    except:
        print("Model not found, creating new one.")
    model.fit(net_in, net_out, epochs=args.epochs, batch_size=args.batch_size, callbacks=callbacks_list)
    model.save(f"models/{args.model}.hdf5")
