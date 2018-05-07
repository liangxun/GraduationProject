import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class DataPreprocess(object):
    """对所有数据归一化到（0,1）"""
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

class DataPreprocess2(object):
    """对每个样本归一化到（均值0，标准差1）"""
    def __init__(self):
        pass

    def lstm_load_multidata(self, filename, seq_len, dim=2, row=1686):
        """LSTM多特征输入"""
        self.mode = 'lstm_dim{}'.format(dim)
        self.dim = dim
        table = pd.read_csv(filename, index_col='Date')
        # 定位收盘价所在列
        self.col_sunspot = table.columns.get_loc('sunspot_ms')
        sequence_length = seq_len + 1
        result = []
        info = []
        """求mean和std时只用到输入部分"""
        for index in range(len(table) - sequence_length):
            data = table[index: index + sequence_length]
            mean = np.mean(data[:-1])
            std = np.std(data[:-1])
            data = (data - mean)/std
            info.append((mean[self.col_sunspot], std[self.col_sunspot]))
            result.append(data.values)
        print("相空间重构后数据集的长度：", len(result))
        result = np.array(result)
        train = result[:row, :]
        print("train set的长度：", row)
        # 输入多维特征，但输出依旧是只有sunspot_ms一维
        x_train = train[:, :-1]
        y_train = train[:, -1, self.col_sunspot]
        x_test = result[row:, :-1]
        y_test = result[row:, -1, self.col_sunspot]
        self.testinfo = info[row:]
        print("test set的长度：", len(y_test))
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], dim))
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], dim))
        print("x_train.shape{}\ty_train.shape{}\nx_test.shape{}\ty_test.shape{}\n".format(
            x_train.shape, y_train.shape, x_test.shape, y_test.shape))
        return [x_train, y_train, x_test, y_test]


    def recover(self, data):
        info = self.testinfo
        for i in range(len(data)):
            data[i] = data[i] * info[i][1] + info[i][0]
        return data

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