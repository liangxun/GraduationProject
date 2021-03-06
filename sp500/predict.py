import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model

from sp500.loader import DataPreprocess2 as DataPreprocess
import EvaluationIndex


def predict_point_by_point(model, data):
    """
    one step ahead    # 直接预测
    Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
    """
    predicted = model.predict(data)
    predicted = np.reshape(predicted, (predicted.size,))
    return predicted


def plot_results_point(predicted_data, true_data,rmse=0):
    fig = plt.figure(facecolor='white')
    plt.plot(true_data, 'o-', label='True Data')
    plt.plot(predicted_data, 'x-', label='Prediction')
    plt.title('the result of lstm\nRMSE={:.2f}'.format(rmse))
    plt.legend()
    plt.show()


if __name__ == "__main__":
    modelpath = './result/lstm_1_dim4_epoch150_steps10_RMSE=16.97.h5'
    import re
    timesteps = int(re.findall('steps(\d*)', modelpath)[0])
    input_dim = int(re.findall('dim(\d*)', modelpath)[0])
    datapath = '../dataset/sp2005_dim{}.csv'.format(input_dim)

    print('> Loading model...')
    model = load_model(modelpath)
    print(model.summary())

    print('> Loading data... ')
    DataLoader = DataPreprocess()
    x_train, y_train, x_test, y_test = DataLoader.lstm_load_multidata(filename=datapath, seq_len=timesteps, dim=input_dim)

    print('>predict...')
    predictions = predict_point_by_point(model, x_test)
    predictions = DataLoader.recover(predictions)
    y_test = DataLoader.recover(y_test)

    eI = EvaluationIndex.evalueationIndex(predictions, y_test)
    print("MSE={}\nRMSE={}\nMAPE={}".format(eI.MSE, eI.RMSE, eI.MAPE))
    plot_results_point(predictions, y_test, eI.RMSE)
    eI.plot_e()
    eI.plot_ae()
    eI.plot_ape()
    eI.correlation()

