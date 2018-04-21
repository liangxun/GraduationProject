import time
from keras.layers.core import Dense, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import matplotlib.pyplot as plt

from sunspot.loader import DataPreprocess
from sunspot.conf import sunspot_ms_path, save_model_path
from sunspot.lstm_predict import predict_point_by_point,plot_results_point
import EvaluationIndex


def build_model(input_shape, layers):
    model = Sequential()
    model.add(LSTM(
        input_shape=input_shape,
        output_dim=layers[0],
        return_sequences=True,
    ))
    model.add(Dropout(0.2))
    model.add(LSTM(
        layers[1],
        return_sequences=False,
    ))
    model.add(Dense(
        layers[2],
        activation='linear',
    ))

    start = time.time()
    model.compile(loss='mse', optimizer='rmsprop')
    print(model.summary())
    print('compilation time:{}'.format(time.time()-start))
    return model


def single_model(input_shape, layers_out):

    model = Sequential()
    model.add(LSTM(
        input_shape=input_shape,
        output_dim=layers_out[0],
        return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(
        output_dim=layers_out[1], activation="linear"))

    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop")
    print("> Compilation Time : ", time.time() - start)
    return model


# hyperparams
epochs = 50
timesteps = 64


def plot_train(history):
    """打印出训练过程中loss的变化趋势"""
    train_loss = history['loss']
    plt.plot(train_loss, label='train_loss')
    plt.legend()
    plt.show()


# train
global_start_time = time.time()
print('> Loading data... ')
DataLoader = DataPreprocess()
x_train, y_train, x_test, y_test = DataLoader.lstm_load_data(sunspot_ms_path, timesteps, row=1686-(timesteps+1))
print('> Data Loaded. Compiling...')

input_shape = (timesteps, 1)
#layers_output = [64, 256, 1]
#model = build_model(input_shape,layers_output)
layers_output = [128, 1]
model = single_model(input_shape,layers_output)
print(model.summary())
hist = model.fit(x_train, y_train, batch_size=128, epochs=epochs, shuffle=True)

print('Training duration (s) : ', time.time() - global_start_time)
plot_train(hist.history)
model.save(save_model_path.format(epochs, timesteps))

predictions = predict_point_by_point(model, x_train)
plt.plot(predictions,label='predict')
plt.plot(y_train,label='true_data')
plt.show()

predictions = predict_point_by_point(model, x_test)
predictions = DataLoader.recover(predictions)
y_test = DataLoader.recover(y_test)
eI = EvaluationIndex.evalueationIndex(predictions, y_test)
print("MSE:", eI.MSE)
print("RMSE:", eI.RMSE)
plot_results_point(predictions, y_test, eI.RMSE)
eI.plot_e()