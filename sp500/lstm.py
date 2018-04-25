import os
import time
import warnings
import matplotlib.pyplot as plt
from keras.layers.core import Dense, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential

from sp500.conf import sp500_datapath, model_path
from sp500 import loader
from sp500 import predict
import EvaluationIndex

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Hide messy TensorFlow warnings
warnings.filterwarnings("ignore")   # Hide messy Numpy warnings


def build_model(input_shape, layers_out):
    model = Sequential()
    model.add(LSTM(
        input_shape=input_shape,
        output_dim=layers_out[0],
        return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(
        layers_out[1],
        return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(
        output_dim=layers_out[2],
        activation='linear'))

    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop")
    print("> Compilation Time : ", time.time() - start)
    return model


def single_model(input_shape, layers_out):
    model = Sequential()
    model.add(LSTM(
        input_shape=input_shape,
        output_dim=layers_out[0],
        return_sequences=False
    ))
    model.add(Dropout(0.1))

    model.add(Dense(
        output_dim=layers_out[1],
        activation='linear'
    ))

    start = time.time()
    model.compile(loss='mse',optimizer='rmsprop')
    print('>compilation time:',time.time()-start)
    return model


def plot_train(history):
    """打印出训练过程中loss的变化趋势"""
    train_loss = history['loss']
    plt.plot(train_loss, label='train_loss')
    plt.legend()
    plt.show()


# hyperparams
epochs = 10
timesteps = 12
lr = 0.001
dropout = 0
batchsize = 256
input_dim = 5
input_shape = (timesteps, input_dim)
layers_output = [32, 1]

if __name__ == '__main__':
    global_start_time = time.time()

    print('> Loading data... ')
    DataLoader = loader.DataPreprocess()
    x_train, y_train, x_test, y_test = DataLoader.lstm_load_multidata(sp500_datapath,timesteps,dim=input_dim)

    print('> Data Loaded. Model Compiling...')
    model = single_model(input_shape, layers_output)
    print(model.summary())

    hist = model.fit(
        x_train,
        y_train,
        batch_size=512,
        nb_epoch=epochs,
        shuffle=False,
        validation_split=0.05)
    print('Training duration (s) : ', time.time() - global_start_time)
    plot_train(hist.history)

    print("> Train finished. save model...")
    #model.save(model_path.format(epochs,timesteps))

    predictions = predict.predict_point_by_point(model, x_train)
    plt.plot(predictions,label='predict')
    plt.plot(y_train,label='true_data')
    plt.show()

    print("> predict...")
    predictions = predict.predict_point_by_point(model, x_test)
    predictions = DataLoader.recover(predictions)
    y_test = DataLoader.recover(y_test)
    eI = EvaluationIndex.evalueationIndex(predictions, y_test)
    print("MSE={}\nRMSE={}\nMAPE={}".format(eI.MSE,eI.RMSE,eI.MAPE))
    predict.plot_results_point(predictions, y_test, eI.RMSE)
    eI.plot_ae()
    eI.plot_e()
    eI.plot_ape()
