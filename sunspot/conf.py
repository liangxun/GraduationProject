sunspot_ms_path = '../dataset/sunspot_ms.csv'

#arma
arma_filepath = sunspot_ms_path

#svm
kernel = 'rbf'

#lstm
lstm_filepath = sunspot_ms_path
epochs = 100
time_steps = 12
lr = 0.01
dropout = 0.2
print_interval = 1000
save_model_path = './lstm_ms_epoch{}_seq{}.h5'.format(epochs,time_steps)

