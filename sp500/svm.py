from sklearn.svm import SVR,LinearSVR
import matplotlib.pyplot as plt

from sp500 import loader
import EvaluationIndex

# hyperparams
seq_len = 10
filename = '../dataset/sp2005_dim1.csv'

#train
DataLoader = loader.DataPreprocess()
x_train, y_train, x_test, y_test = DataLoader.svm_load_data(filename,seq_len)
svr = SVR(kernel='rbf', epsilon=0.00001, C=100)
svr.fit(x_train, y_train)
"""
svr = LinearSVR(epsilon=0.00001, C=1.0)
svr.fit(x_train, y_train)
"""

#predict
prediction = svr.predict(x_test)

y_test = DataLoader.recover(y_test)
prediction = DataLoader.recover(prediction)

eI = EvaluationIndex.evalueationIndex(prediction,y_test)
print('MSE={}\nRMSE={}\nMAPE={}'.format(eI.MSE,eI.RMSE,eI.MAPE))
plt.plot(y_test, label='true_data')
plt.plot(prediction, label='predict_data')
plt.title('the result of svr\nRMSE={:.2f}'.format(eI.RMSE))
plt.legend()
plt.show()
"""
eI.plot_ae()
eI.plot_e()
eI.plot_ape()
"""