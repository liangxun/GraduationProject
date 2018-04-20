import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class DataPreprocess(object):
    def __init__(self):
        self.mode = None

    def arma_load_data(self, filename,normalise_bool=True,row=1686):
        self.mode = 'arma'
        table = pd.read_csv(filename)
        data = table['sunspot_ms']
        #show(data)
        data = np.array(data)
        print('len(data)', len(data))

        if normalise_bool is True:
            self.max = max(data)
            self.min = min(data)
            data = np.array(self.normalise(data))

        train_data = data[:row]
        print("train_data的长度:", len(train_data))
        test_data = data[row:]
        print("test_data的长度:", len(test_data))
        return data, (train_data, test_data)

    def svm_load_data(self,filename, seq_len, normalise_bool=True,row=2500):
        self.mode = 'svm'
        table = pd.read_csv(filename)
        #data = table['sunspot.month']
        data = table['sunspot_ms']
        #show(data)
        data = np.array(data)
        print('len(data)',len(data))

        if normalise_bool is True:
            self.max = max(data)
            self.min = min(data)
            data = self.normalise(data)

        sequence_length = seq_len + 1
        result = []
        for index in range(len(data) - sequence_length):
            result.append(data[index: index + sequence_length])
        print("相空间重构后数据集的长度：", len(result))
        result = np.array(result)
        train = result[:row, :]
        print("train set的长度：", row)
        np.random.shuffle(train)
        x_train = train[:, :-1]
        y_train = train[:, -1]
        x_test = result[row:, :-1]
        y_test = result[row:, -1]
        print("test set的长度：", len(y_test))

        return [x_train, y_train, x_test, y_test]

    def lstm_load_data(self, filename, seq_len, normalise_bool=True,row=2500):
        self.mode = 'lstm'
        table = pd.read_csv(filename)
        data = table['sunspot_ms']
        data = np.array(data)
        print('len(data)', len(data))

        if normalise_bool is True:
            self.max = max(data)
            self.min = min(data)
            data = self.normalise(data)

        sequence_length = seq_len + 1
        result = []
        for index in range(len(data) - sequence_length):
            result.append(data[index: index + sequence_length])
        print("相空间重构后数据集的长度：",len(result))
        result = np.array(result)
        train = result[:row, :]
        print("train set的长度：", row)
        #np.random.shuffle(train)
        x_train = train[:, :-1]
        y_train = train[:, -1]
        x_test = result[row:, :-1]
        y_test = result[row:, -1]
        print("test set的长度：", len(y_test))
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        return [x_train, y_train, x_test, y_test]

    def normalise(self, data):
        normalised_data = [(p-self.min)/(self.max-self.min) for p in data]
        #show(normalised_data)
        return normalised_data

    def recover(self, data):
        print('DataLoader.mode={}'.format(self.mode))
        recovered_data = [p * (self.max-self.min) + self.min for p in data]
        return recovered_data

def show(data):
    plt.figure()
    plt.plot(data)
    plt.show()

if __name__ == '__main__':
    filename = '../dataset/sunspot_ms.csv'
    row = 1686
    DataLoader = DataPreprocess()
    #data, (train_data, test_data) = DataLoader.arma_load_data()
    #x_train, y_train, x_test, y_test = DataLoader.lstm_load_data(filename,12,row=1686)
    x_train, y_train, x_test, y_test = DataLoader.svm_load_data(filename, 12, row=1686)
    show(DataLoader.recover(y_test))