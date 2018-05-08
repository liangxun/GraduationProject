import numpy as np
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 让图例正常显示中文
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


class evalueationIndex(object):
    """
    预测精度评价指标
    """
    def __init__(self,prediction, true_data):
        true_data = np.array(true_data)
        prediction = np.array(prediction)
        if(true_data.size != prediction.size):
            print('error: true_data.size != prediction.size')
        obj_size = true_data.size

        # ----这四个指标是list，可以plt
        # 预测误差 e
        self.e = true_data-prediction
        # 绝对预测误差
        self.ae = self.e.__abs__()
        # 百分预测误差 pe
        self.pe = self.e/true_data * 100
        # 百分绝对预测误差
        self.ape = self.pe.__abs__()

        obj_size = true_data.size
        # ----下面的指标是一个数值
        #平均误差 ME
        self.ME = self.e.sum()/obj_size
        #平均绝对误差 MAE
        self.MAE = self.ae.sum()/obj_size
        #均方误差 MSE
        self.MSE = (self.e**2).sum()/obj_size
        #均方根误差 RMSE
        self.RMSE = np.sqrt(self.MSE)
        # 平均绝对百分误差
        self.MAPE = self.ape.sum()/obj_size


    def plot_ae(self):
        plt.plot(self.ae, label="绝对预测误差")
        plt.legend()
        plt.show()

    def plot_e(self):
        plt.plot(self.e, label='预测误差')
        plt.legend()
        plt.show()

    def plot_ape(self):
        plt.plot(self.ape,label='百分绝对预测误差')
        plt.legend()
        plt.show()

    # 计算残差的自相关函数和偏自相关函数
    def correlation(self):
        plot_acf(self.e, lags=50)
        plot_pacf(self.e, lags=50)
        plt.show()