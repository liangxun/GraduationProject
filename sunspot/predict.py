import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model

import EvaluationIndex
# from sunspot.loader import DataPreprocess
from sunspot import load_demo


def predict_point_by_point(model, data):
    """
    one step ahead    # 直接预测
    Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
    """
    predicted = model.predict(data)
    predicted = np.reshape(predicted, (predicted.size,))
    return predicted


def plot_results_point(predicted_data, true_data,RMSE):
    plt.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.title('the result of lstm\nRMSE = {:.2f}'.format(RMSE))
    plt.legend()
    plt.show()


if __name__ == '__main__':
    modelpath = './result/lstm1_dim2_epoch100_steps10_RMSE=1.87.h5'
    import re
    timesteps = int(re.findall('steps(\d*)', modelpath)[0])
    input_dim = int(re.findall('dim(\d*)', modelpath)[0])
    datapath = '../dataset/sunspot_ms_dim{}.csv'.format(input_dim)

    print('> Loading model...')
    model = load_model(modelpath)
    print(model.summary())
    print('> Loading data... ')
    DataLoader = load_demo.DataPreprocess()
    x_train, y_train, x_test, y_test = DataLoader.lstm_load_multidata(filename=datapath, seq_len=timesteps, dim=input_dim, row=1686-(timesteps+1))
    print('>predict...')
    predictions = predict_point_by_point(model, x_test)
    predictions = DataLoader.recover(predictions)
    y_test = DataLoader.recover(y_test)
    eI = EvaluationIndex.evalueationIndex(predictions, y_test)
    print("MSE={}\nRMSE={}\nMAPE={}".format(eI.MSE, eI.RMSE, eI.MAPE))
    plot_results_point(predictions, y_test, eI.RMSE)
    e = eI.e
    eI.plot_e()
    eI.plot_ae()
    eI.plot_ape()


