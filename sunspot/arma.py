import numpy as np
import pyflux as pf
import matplotlib.pyplot as plt

from loader import DataPreprocess
import EvaluationIndex

#hyerparams
ar = 12
ma = 2
sunspot_ms_path = '../dataset/sunspot_ms.csv'

# load data
print(">load data...")
DataLoader = DataPreprocess()
data, (train_data, test_data) = DataLoader.arma_load_data(sunspot_ms_path)

model = pf.ARIMA(data=data, ar=ar, ma=ma, family=pf.Normal())
prediction = np.squeeze(np.array(model.predict_is(h=len(test_data), fit_once=True, fit_method='MLE')))

prediction = DataLoader.recover(prediction)
test_data = DataLoader.recover(test_data)
eI = EvaluationIndex.evalueationIndex(prediction, test_data)
print('MSE={}\nRMSE={}'.format(eI.MSE, eI.RMSE))
plt.plot(test_data, label='true_data')
plt.plot(prediction, label='predictions')
plt.title('the result of arma\nRMSE={:.2f}'.format(eI.RMSE))
plt.legend()
plt.show()
eI.plot_e()