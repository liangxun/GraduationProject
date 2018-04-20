import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']  #让图例正常显示中文
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号


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
        #预测误差 e
        self.e = true_data-prediction
        #绝对预测误差，可以输出查看plot_ae()
        self.ae = self.e.__abs__()
        
        #平均误差 ME
        self.ME = self.e.sum()/obj_size
        #平均绝对误差 MAE
        self.MAE = self.ae.sum()/obj_size
        #均方误差 MSE
        self.MSE = (self.e**2).sum()/obj_size
        #均方根误差 RMSE
        self.RMSE = np.sqrt(self.MSE)

    def plot_ae(self):
        plt.plot(self.ae, label="预测绝对误差")
        plt.legend()
        plt.show()

    def plot_e(self):
        plt.plot(self.e, label='预测误差')
        plt.legend()
        plt.show()