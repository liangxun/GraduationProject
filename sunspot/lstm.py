import time
import matplotlib.pyplot as plt

from sunspot.load_demo import DataPreprocess
from sunspot import lstm_model
from sunspot.predict import predict_point_by_point, plot_results_point
import EvaluationIndex


def plot_train(history):
    """打印出训练过程中loss的变化趋势"""
    train_loss = history['loss']
    plt.plot(train_loss, label='train_loss')
    plt.legend()
    plt.show()


# hyperparams
epochs = 100
timesteps = 42
lr = 0.001
dropout = 0.001
batchsize = 128
input_dim = 2

layers_output = [64, 1]
input_shape = (timesteps, input_dim)
filename = '../dataset/sunspot_ms_dim{}.csv'.format(input_dim)
model_path = './result/{}_dim{}_epoch{}_steps{}_RMSE={:.2f}.h5'

if __name__ == '__main__':
    global_start_time = time.time()

    print('> Loading data... ')
    DataLoader = DataPreprocess()
    x_train, y_train, x_test, y_test = DataLoader.lstm_load_multidata(filename, timesteps, dim=input_dim, row=1686-(timesteps+1))
    print('> Data Loaded. Compiling...')
    model, model_name = lstm_model.lstm1(input_shape, layers_output,lr=lr,dropout=dropout)
    print(model.summary())
    hist = model.fit(
        x_train,
        y_train,
        batch_size=batchsize,
        epochs=epochs,
        shuffle=True,
        validation_split=0)
    print('Training duration (s) : ', time.time() - global_start_time)
    plot_train(hist.history)

    # 在训练集上预测，看是否欠拟合
    predictions = predict_point_by_point(model, x_train)
    plt.plot(predictions, label='predict')
    plt.plot(y_train, label='true_data')
    plt.legend()
    plt.show()

    # 在测试集上预测，计算测量指标
    print("> predict...")
    predictions = predict_point_by_point(model, x_test)
    predictions = DataLoader.recover(predictions)
    y_test = DataLoader.recover(y_test)
    eI = EvaluationIndex.evalueationIndex(predictions, y_test)
    print("MSE={}\nRMSE={}\nMAPE={}".format(eI.MSE, eI.RMSE, eI.MAPE))
    plot_results_point(predictions, y_test, eI.RMSE)
    eI.plot_ae()
    eI.plot_e()
    eI.plot_ape()

    print("> Train finished. save model...")
    model.save(model_path.format(model_name, input_dim, epochs, timesteps, eI.RMSE))