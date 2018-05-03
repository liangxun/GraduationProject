from sklearn.svm import SVR,LinearSVR
import matplotlib.pyplot as plt

from loader import DataPreprocess
import EvaluationIndex


# hyperparams
input_dim = 12
kernel = 'rbf'
sunspot_ms_path = '../dataset/sunspot_ms.csv'

# load data
print(">load data...")
DataLoader = DataPreprocess()
x_train, y_train, x_test, y_test = DataLoader.svm_load_data(sunspot_ms_path, seq_len=input_dim, row=1686-input_dim-1)

# train
print(">train model...")
#svr = SVR(kernel=kernel, epsilon=0, C=1)
#svr.fit(x_train, y_train)
svr = LinearSVR(epsilon=0.00001, C=5000)
svr.fit(x_train, y_train)

# predict
print(">predict...")
y_predict = svr.predict(x_test)
y_predict = DataLoader.recover(y_predict)
y_test = DataLoader.recover(y_test)
eI = EvaluationIndex.evalueationIndex(y_predict, y_test)
print('MSE={}\nRMSE={}'.format(eI.MSE, eI.RMSE))
plt.plot(y_test, label='true_data')
plt.plot(y_predict, label='predict_data')
plt.title('the result of svr\nRMSE={:.2f}    MAPE={:.2f}'.format(eI.RMSE,eI.MAPE))
plt.legend()
plt.show()