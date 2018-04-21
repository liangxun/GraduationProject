import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model

import EvaluationIndex
from sunspot.loader import DataPreprocess
from sunspot.conf import sunspot_ms_path,time_steps


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
    model = load_model('./result/lstm_ms_epoch25_steps50.h5')
    print(model.summary())
    DataLoader = DataPreprocess()
    x_train,y_train,x_test,y_test = DataLoader.lstm_load_data(sunspot_ms_path,time_steps,row=1686-(time_steps+1))

    predict = predict_point_by_point(model,x_test)
    predict = DataLoader.recover(predict)
    y_test = DataLoader.recover(y_test)
    eI = EvaluationIndex.evalueationIndex(predict,y_test)
    print("MSE:", eI.MSE)
    print("RMSE:", eI.RMSE)
    plot_results_point(predict, y_test, eI.RMSE)
    eI.plot_ae()


