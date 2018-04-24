#太阳黑子数据集
sunspot_ms.csv
    1855年12月——2017年9月的平滑月均值
sunspot_ms_diff.csv
    在sunspot_ms.csv的基础上新增加一列：差分diff，可以表示波动大小，构成二维的输入（黑子数，波动）
    
    
#标普500数据集
GSPC2005-2015.csv   
    从雅虎财经下载的原始数据 
sp2005-2015.csv 
    前期使用的数据，在GSPC2005-2015中去掉了[adj close]
spv2.csv
    新增了[High-Low][close-preclose]两列
    输入特征 = （开盘价，最高价，最低价，收盘价，成交量，振幅，涨幅）
    其中振幅=最高价 - 最低价；
        涨幅=收盘价 - 前一天的收盘价