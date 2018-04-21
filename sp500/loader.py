import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class DataPreprocess(object):
    def __init__(self):
        pass

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

    def lstm_load_multidata(self, filename,seq_len,dim=1,normalise_bool=True,row=2000):
        pass

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
        normalised_data = (data-self.min)/(self.max-self.min)
        return normalised_data

    def unnormalise(self, data):
        unnormalised_data = data * (self.max-self.min) + self.min
        return unnormalised_data

    def recover(self, data):
        if self.normalise_bool is True:
            data = self.unnormalise(data)
        recovered_data = np.exp(data)
        return recovered_data


def show(data,label='data'):
    plt.figure()
    plt.plot(data,label=label)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    datapath = '../dataset/sp2005-2015.csv'
    print('> Loading data... ')
    DataLoader = DataPreprocess()
    #x_train, y_train, x_test, y_test = DataLoader.lstm_load_data(datapath, 50)
    #data, (train,test) = DataLoader.arima_load_data(datapath,50)
    x_train, y_train, x_test, y_test = DataLoader.svm_load_data(datapath, 50)
    show(y_train)
    show(y_test)
    show(DataLoader.recover(y_test))