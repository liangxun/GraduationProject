import pyflux as pf
import numpy as np
import matplotlib.pyplot as plt

from sp500 import loader
import EvaluationIndex

filename = '../dataset/sp2005-2015.csv'
seq_len = 50

DataLoader = loader.DataPreprocess()
data, (train_data, test_data) = DataLoader.arima_load_data(filename, seq_len)
model = pf.ARIMA(data=data, ar=12, ma=2, family=pf.Normal())
prediction = np.squeeze(np.array(model.predict_is(h=len(test_data), fit_once=True, fit_method='MLE')))

test_data = DataLoader.recover(test_data)
prediction = DataLoader.recover(prediction)

eI = EvaluationIndex.evalueationIndex(prediction, test_data)
print('MSE={}\nRMSE={}\nMAPE={}'.format(eI.MSE,eI.RMSE,eI.MAPE))

plt.plot(test_data, label='true_data')
plt.plot(prediction, label='predictions')
plt.title('the result of arima\nRMSE={:.2f}'.format(eI.RMSE))
plt.legend()
plt.show()
eI.plot_ae()
eI.plot_e()
eI.plot_ape()