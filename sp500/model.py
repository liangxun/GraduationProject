from keras.layers.core import Dense, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras import optimizers
from keras.utils import plot_model
import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Hide messy TensorFlow warnings
warnings.filterwarnings("ignore")   # Hide messy Numpy warnings


def lstm_2(input_shape, layers_out, lr=0.001, dropout=0.1):
    model_name = 'lstm2'
    model = Sequential()
    model.add(LSTM(
        input_shape=input_shape,
        output_dim=layers_out[0],
        return_sequences=True))
    model.add(Dropout(dropout))

    model.add(LSTM(
        layers_out[1],
        return_sequences=False))
    model.add(Dropout(dropout))

    model.add(Dense(
        output_dim=layers_out[2],
        activation='linear'))
    model.compile(loss="mse", optimizer=optimizers.rmsprop(lr=lr))
    return model, model_name


def lstm_1(input_shape, layers_out, lr=0.001, dropout=0.1):
    model_name = 'lstm_1'
    model = Sequential()
    model.add(LSTM(
        input_shape=input_shape,
        output_dim=layers_out[0],
        return_sequences=False
    ))
    model.add(Dropout(dropout))

    model.add(Dense(
        output_dim=layers_out[1],
        activation='linear'
    ))

    model.compile(loss='mae', optimizer=optimizers.rmsprop(lr=lr))
    return model, model_name


def lstm_bp(input_shape,layers_out,lr=0.001,dropout=0.01):
    model_name = 'lstm_bp'
    model = Sequential()
    model.add(LSTM(layers_out[0], input_shape=input_shape, dropout=dropout,return_sequences=True))
    model.add(LSTM(layers_out[1], dropout=dropout, return_sequences=False))
    model.add(Dense(layers_out[2], activation='relu'))
    model.add(Dense(layers_out[3]))
    model.compile(optimizer=optimizers.rmsprop(lr=lr), loss='mse')
    return model, model_name


if __name__ == '__main__':
    layer_out = [32, 64, 32, 1]
    input_shape = (12, 7)
    model,model_name= lstm_bp(input_shape, layer_out)
    print(model.summary())
    #plot_model(model, show_shapes=True, show_layer_names=False)
    plot_model(model, to_file='./model.png', show_shapes=True, show_layer_names=False)
