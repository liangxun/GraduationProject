import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class DataPreprocess(object):
    def __init__(self):
        self.dim = 1
        self.normalise_bool = True

    def arima_load_data(self,filename,seq_len,normalise_bool=True,row=2000):
        table = pd.read_csv(filename)
        table.index = pd.to_datetime(table.Date)
        data = table['Close']
        data = pd.Series(np.log(data))

        if normalise_bool is True:
            self.normalise_bool = normalise_bool
            self.max = max(data)
            self.min = min(data)
            data = np.array(self.normalise(data))

        data = np.array(data)
        row += seq_len
        train_data = data[:row]
        print("train_data的长度:", len(train_data))
        test_data = data[row:]
        print("test_data的长度:", len(test_data))
        return data, (train_data, test_data)

    def svm_load_data(self,filename, seq_len, normalise_bool=True, row=2000):
        table = pd.read_csv(filename)
        table.index = pd.to_datetime(table.Date)
        data = table['Close']
        data = pd.Series(np.log(data))
        if normalise_bool is True:
            self.normalise_bool = normalise_bool
            self.max = max(data)
            self.min = min(data)
            data = np.array(self.normalise(data))

        data = np.array(data)
        sequence_length = seq_len + 1
        result = []
        for index in range(len(data) - sequence_length):
            result.append(data[index: index + sequence_length])
        result = np.array(result)
        print("相空间重构sp500后result的维度：",result.shape)
        train = result[:row]
        #np.random.shuffle(train)
        x_train = train[:, :-1]
        y_train = train[:, -1]
        x_test = result[row:, :-1]
        y_test = result[row:, -1]

        print("x_train.shape={}\ty_train.shape={}\nx_test.shape={}\ty_test.shape={}\n".format(
            x_train.shape, y_train.shape, x_test.shape, y_test.shape))
        return [x_train, y_train, x_test, y_test]

    def lstm_load_multidata(self, filename, seq_len, dim=2, normalise_bool=True, row=2000):
        """LSTM多特征输入"""
        self.mode = 'lstm_dim{}'.format(dim)
        self.dim = dim
        table = pd.read_csv(filename,index_col='Date')
        #data = np.log(table)
        data = table
        if normalise_bool is True:
            self.max = data.max()
            self.min = data.min()
            self.col_close = data.columns.get_loc('Close')
            data = self.normalise(data)
        data = np.array(data)
        sequence_length = seq_len + 1
        result = []
        for index in range(len(data) - sequence_length):
            result.append(data[index: index + sequence_length])
        print("相空间重构后数据集的长度：", len(result))
        result = np.array(result)
        train = result[:row, :]
        print("train set的长度：", row)
        #输入多维特征，但输出依旧是只有close一维
        x_train = train[:, :-1]
        y_train = train[:, -1, self.col_close]
        x_test = result[row:, :-1]
        y_test = result[row:, -1, self.col_close]
        print("test set的长度：", len(y_test))
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], dim))
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], dim))
        print("x_train.shape{}\ty_train.shape{}\nx_test.shape{}\ty_test.shape{}\n".format(
            x_train.shape, y_train.shape, x_test.shape, y_test.shape))
        return [x_train, y_train, x_test, y_test]

    def lstm_load_data(self, filename,seq_len,dim=1,normalise_bool=True,row=2000):
        self.input_dim = dim
        table = pd.read_csv(filename)
        table.index = pd.to_datetime(table.Date)
        data = table['Close']
        data = pd.Series(np.log(data))

        # normalise range of data to (0,1)
        if normalise_bool is True:
            self.normalise_bool = normalise_bool
            self.max = data.max()
            self.min = data.min()
            data = self.normalise(data)

        data = np.array(data)
        sequence_length = seq_len + 1
        result = []
        for index in range(len(data) - sequence_length):
            result.append(data[index: index + sequence_length])
        result = np.array(result)
        print("相空间重构后 data.shape：", result.shape)

        train = result[:row]
        #np.random.shuffle(train)
        x_train = train[:, :-1]
        y_train = train[:, -1]
        x_test = result[row:, :-1]
        y_test = result[row:, -1]

        x_train = x_train[:, :, np.newaxis]
        x_test = x_test[:, :, np.newaxis]

        print("x_train.shape{}\ty_train.shape{}\nx_train.shape{}\ty_train.shape{}\n".format(
            x_train.shape, y_train.shape, x_test.shape, y_test.shape))

        return [x_train, y_train, x_test, y_test]

    def normalise(self, data):
        """将所有列的数据都归一化到（0,1）"""
        normalised_data = (data-self.min)/(self.max-self.min)
        return normalised_data

    def unnormalise(self, data):
        """
        只需要恢复close这一列。
        当lstm的输入维度大于2时，max保存了所有列的最大值，需要指定Close所在列的下标"""
        if self.dim is 1:
            recovered_data = data * (self.max-self.min) + self.min
        else:
            recovered_data = data*(self.max[self.col_close] - self.min[self.col_close]) + self.min[self.col_close]
        return recovered_data

    def recover(self, data):
        if self.normalise_bool is True:
            data = self.unnormalise(data)
        # recovered_data = np.exp(data)
        recovered_data = data
        return recovered_data


def show(data,label='data'):
    plt.figure()
    plt.plot(data, label=label)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    dim = 4
    datapath = '../dataset/sp2009_dim{}.csv'.format(dim)
    print('> Loading data... ')
    DataLoader = DataPreprocess()
    #x_train, y_train, x_test, y_test = DataLoader.lstm_load_data(datapath, 50)
    #data, (train,test) = DataLoader.arima_load_data(datapath,50)
    x_train, y_train, x_test, y_test = DataLoader.lstm_load_multidata(datapath,50,dim)
    show(y_train)
    show(y_test)
    show(DataLoader.recover(y_test))