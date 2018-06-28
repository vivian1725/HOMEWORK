
# 主題： 比幣特價格預測

![Alt text](bitcoin.jpg)
--------

# 動機：

### 這幾年加密貨幣興起，幣價大起大落，上沖下洗，這樣波動幅度巨大的交易標的一項吸引許多投(ㄉㄨˇ)資(ㄊㄨˊ)人的關注。
### 如果能利用機器學習的方式去預估給一個上漲或是下跌的波段，將帶來巨大的財富。

### 所以這次期末報告就決定嘗試利用不同的方法來預估比特幣的價格，再來看是否能有效地預測。

-------

# 計畫摘要：

-------

#### 1.利用網路爬蟲技術取得比特幣歷史價格
#### 2.使用深度學習預測
#### 3.建構LINE BOT 機器人取得預測結果


# 實作部份：
-------
#### 1. 建構LINEBOT 機器人放至Heroku 可查詢當日比特幣價格
#### 2. 使用程式捉取Quandl各交易所的比特幣平均金額
#### 3. 再捉取 Poloniex 上不同的虛擬幣值一起放入
#### 4. 使用時間序列預測的LSTM神經網路預測價格


===================

## 建一個LINEBOT 機器人可以查詢目前比特幣價格

#### app.py / Procfile / requirements.txt
--------

###  申請 LINE Messaging API：

#### 利用原來LINEID 申請一個 Messaging API 先
#### 需要 ISSU Channel secret／Channel access token (long-lived)  之後程式會用到需要先記下來

![Alt text](LINE1.png)

###  申請 Heroku 將程式佈署上去：

#### 先申請帳號
#### 並建立一個 APP，自己命名

![Alt text](LINE2.png)

#### 把已命名的APP回填至 申請的Messaging API webHook裡

![Alt text](LINE3.png)
-------

### 程式說明：
-------
####  requirements.txt ： 需要安裝的元件
####  app.py ： 1.裡頭需對應 Messaging API給的 Channel secret／Channel access token (long-lived)
![Alt text](LINE4.png)
####                   2.捉取比特幣今日價格，若使用者有詢問時可回覆其價格
####                   3.若使用者有詢問時，可回覆imgur上產生好的比特幣價格trend chart

                 
#### 佈署至Heroku 方法 ：
####                                    1.需先下載Heroku CLI 
####                                    2.確認Local程式位置後
####                                    3.cmd 下指令 （git add. git push heroku master)                     

![Alt text](LINE5.png)

===============

## 使用套件

  * json
  * requests
  * pandas 
  * matplotlib
  * numpy 
  * pickle
  * quandl
  * keras 
  

# 使用的套件
#### 這次主要是使用Keras作為LSTM模型建立的套件
#### 並輔以numpy與pandas做資料處理
#### 再利用matplotlib做繪圖
#### 資料來源的部分有使用到資料來源專用的接口套件 quandl


```python
import json
import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle 
import quandl
import codecs
from datetime import datetime
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import Activation
```

    Using TensorFlow backend.


# 定義函數

## 下載來自Quandl的 Bitcoin 資料集
#### Quandl本身提供了Python的資料擷取套件
#### 所以只需要簡單的設定就可以輕鬆地獲得資料
#### 中間利用pickle作資料的備份
#### api_key可以不用填沒問題，但會有連接上線
#### 所以還是建議到Quandl申請免費的帳號獲取API KEY


```python
def get_quandl_data(quandl_id):
    '''Download and cache Quandl dataseries''' 
    cache_path = '{}.pkl'.format(quandl_id).replace('/','-') 
    
    #Enter quandl API Key
    quandl.ApiConfig.api_key = "zQm4uFHeJru86SyaLs6v"
    
    print('Downloading {} from Quandl'.format(quandl_id)) 
    df = quandl.get(quandl_id, returns="pandas") 
    df.to_pickle(cache_path) 
    print('Cached {} at {}'.format(quandl_id, cache_path)) 
    return df
```

## 從不同的DataFrame萃取出特定的欄位合併成新的DataFrame


```python
def merge_dfs_on_column(dataframes, labels, col):
    '''Merge a single column of each dataframe into a new combined dataframe''' 
    series_dict = {} 
    for index in range(len(dataframes)): 
        series_dict[labels[index]] = dataframes[index][col] 
    return pd.DataFrame(series_dict)
```

## 從Poloniex抓取更多其他虛擬貨幣的價格


### 讀取JSON檔
#### 藉由Poloniex提供的Web service來抓去幣價的JSON檔
#### 所以先建立一個函式用抓取JSON


```python
def get_json_data(json_url, cache_path):
    '''Download and cache JSON data, return as a dataframe.''' 
    print('Downloading {}'.format(json_url)) 
    #df = pd.read_json(codecs.open(json_url,'r','utf-8'))
    json=requests.get(json_url, verify=True).text
    df = pd.read_json(json) 
    df.to_pickle(cache_path) 
    print('Cached {} at {}'.format(json_url, cache_path)) 
    return df
```

### Poloniex抓取資料
#### 實際利用上面的函式來抓去JSON檔
#### 再利用pandas儲存成DataFrame的格式方便後續的操作


```python
def get_crypto_data(poloniex_pair): 
    base_polo_url = 'https://poloniex.com/public?command=returnChartData&currencyPair={}&start={}&end={}&period={}' 
    start_date = datetime.strptime('2016-01-01', '%Y-%m-%d') # get data from the start of 2016
    end_date = datetime.now() # up until today
    pediod = 86400 # pull daily data (86,400 seconds per day) 
    '''Retrieve cryptocurrency data from poloniex''' 
    json_url = base_polo_url.format(poloniex_pair, start_date.timestamp(), end_date.timestamp(), pediod) 
    data_df = get_json_data(json_url, poloniex_pair) 
    data_df = data_df.set_index('date')
    return data_df 
```

## 資料分集
#### 把資料其切分成訓練集與測試集
#### 這邊的設定是抓90%數據訓練，10%數據測試


```python
def train_test_split(df, test_size=0.1):
    split_row = len(df) - int(test_size * len(df))
    train_data = df.iloc[:split_row]
    test_data = df.iloc[split_row:]
    return train_data, test_data
```

## 繪圖


```python
def line_plot_s(line1, label1, title):
    fig, ax = plt.subplots(1, figsize=(16, 9))
    ax.plot(line1, label=label1, linewidth=2)
    ax.set_ylabel('price [USD]', fontsize=14)
    ax.set_title(title, fontsize=18)
    ax.legend(loc='best', fontsize=18)
```


```python
def line_plot(line1, line2, label1=None, label2=None, title=''):
    fig, ax = plt.subplots(1, figsize=(16, 9))
    ax.plot(line1, label=label1, linewidth=2)
    ax.plot(line2, label=label2, linewidth=2)
    ax.set_ylabel('price [USD]', fontsize=14)
    ax.set_title(title, fontsize=18)
    ax.legend(loc='best', fontsize=18)
```

## 對資料集做normailise的函數
#### 因為後續不是只有BTC的價錢，還有加入其他的虛擬貨幣一起
#### 所以為了避免受不同幣種幣價原本的高低影響，所以做normailise
#### 那這邊的做法是讓價錢變成跟第一天的價錢的漲跌幅
#### 也就是說假設第一天價錢是8000，第二天價錢變成10000，第三天價錢又變回8000
#### 則在normailise之後就會變成0、0.25、0
#### 那14天的資料也都是以照這樣的邏輯進行


```python
def normalise_zero_base(df):
    """ Normalise dataframe column-wise to reflect changes with
        respect to first entry.
    """
    return df / df.iloc[0] - 1
```

## 建立函數快速LSTM建模所需的資料模式
#### 預設是14天週期，並使用normailise
#### 假設今天是7/15，透過這函數就會從資料集中收集7/1-7/14共14天的資料並做normailise
#### 在合併到訓練用的模型中


```python
def extract_window_data(df, window=14, zero_base=True):
    """ Convert dataframe to overlapping sequences/windows of
        length `window`.
    """
    window_data = []
    for idx in range(len(df) - window):
        tmp = df[idx: (idx + window)].copy()
        if zero_base:
            tmp = normalise_zero_base(tmp)
        window_data.append(tmp.values)
    return np.array(window_data)
```

## 綜合上述的函式
#### 做到一鍵分割、合併、normailise訓練與測試用的目標與資料集 


```python
def prepare_data(df, window=14, zero_base=True, test_size=0.1):
    """ Prepare data for LSTM. """
    # train test split
    train_data, test_data = train_test_split(df, test_size)
    
    # extract window data
    X_train = extract_window_data(train_data, window, zero_base)
    X_test = extract_window_data(test_data, window, zero_base)
    
    # extract targets
    y_train = train_data.average[window:].values
    y_test = test_data.average[window:].values
    if zero_base:
        y_train = y_train / train_data.average[:-window].values - 1
        y_test = y_test / test_data.average[:-window].values - 1
    return train_data, test_data, X_train, X_test, y_train, y_test
```

## LSTM建模函數
#### 利用Keras內的模型建立我們的LSTM模型


```python
def build_lstm_model(input_data, output_size, neurons=20,
                     activ_func='linear', dropout=0.25,
                     loss='mae', optimizer='adam'):
    model = Sequential()
    model.add(LSTM(neurons, input_shape=(
              input_data.shape[1], input_data.shape[2])))
    model.add(Dropout(dropout))
    model.add(Dense(units=output_size))
    model.add(Activation(activ_func))
    model.compile(loss=loss, optimizer=optimizer)
    return model
```

=======================================================================================================

# 抓取各交易所比特幣交易資料
### 這邊利用Quandl提供的資料集來做為資料集
### Quandl上有持續在更新且以美金報價的交易所為下列4所
### 因為虛擬貨幣的交易在各國的規範日益嚴苛
### 所以能使用法幣出入金的交易所不多


```python
# Pull pricing data form 4 BTC exchanges 
exchanges = ['COINBASE','BITSTAMP','ITBIT','KRAKEN'] 
exchange_data = {} 
for exchange in exchanges: 
    exchange_code = 'BCHARTS/{}USD'.format(exchange) 
    btc_exchange_df = get_quandl_data(exchange_code) 
    exchange_data[exchange] = btc_exchange_df 
```

    Downloading BCHARTS/COINBASEUSD from Quandl
    Cached BCHARTS/COINBASEUSD at BCHARTS-COINBASEUSD.pkl
    Downloading BCHARTS/BITSTAMPUSD from Quandl
    Cached BCHARTS/BITSTAMPUSD at BCHARTS-BITSTAMPUSD.pkl
    Downloading BCHARTS/ITBITUSD from Quandl
    Cached BCHARTS/ITBITUSD at BCHARTS-ITBITUSD.pkl
    Downloading BCHARTS/KRAKENUSD from Quandl
    Cached BCHARTS/KRAKENUSD at BCHARTS-KRAKENUSD.pkl


### 僅捉取2016年之後的加權價格
### 並增加一欄平均值作為訓練的目標價格
### 只使用2016年之後的價格主要是為了配合下面引進其他虛擬貨幣的部分
### 捉取的資料期有部分缺失或為0的資料
### 缺失或為0的資料則利用當天其他交易所的平均價格替補


```python
# merge the  BTC price dataseries' into a single dataframe
btc_usd_datasets = merge_dfs_on_column(list(exchange_data.values()), list(exchange_data.keys()), 'Weighted Price')
# extract data after 2016
btc_usd_datasets = btc_usd_datasets.loc[btc_usd_datasets.index >= '2016-01-01']
# Remove "0" values 
btc_usd_datasets.replace(0, np.nan, inplace=True)
# replace nan with row mean
fill_value = pd.DataFrame({col: btc_usd_datasets.mean(axis=1) for col in btc_usd_datasets.columns})
btc_usd_datasets = btc_usd_datasets.fillna(value=fill_value)
#
btc_usd_datasets['average'] = btc_usd_datasets.mean(axis=1)
```


```python
line_plot_s(btc_usd_datasets.average, 'avage', 'BTC')
```


![png](output_30_0.png)


 ## 並從 poloniex 下載其它虛擬貨幣資料拉入當欄位
 ### 這邊選擇除了BTC以外幾個比較知名的虛擬貨幣
 ### 包含了乙太坊、萊特幣、瑞波幣、門羅幣等
 ### 這4種分別都有各自知名的成因與背後支持的技術
 #### 但因為前面有提到法幣出入金的限制
 #### 所以這邊報價選擇USDT，USDT是一款"號稱"與美金1:1掛鉤的虛擬貨幣
 #### 雖然實際上USDT的價值可能不一定是1美金，但相差不大，姑且作為美金報價


```python
# 從Poloniex下載交易資料 我們將下載4個虛擬貨幣： Ethereum，Litecoin，Ripple，Monero的交易資料
altcoins = ['ETH','LTC','XRP','XMR']
altcoin_data = {}
for altcoin in altcoins:
    coinpair = 'USDT_{}'.format(altcoin)
    crypto_price_df = get_crypto_data(coinpair)
    altcoin_data[altcoin] = crypto_price_df
```

    Downloading https://poloniex.com/public?command=returnChartData&currencyPair=USDT_ETH&start=1451577600.0&end=1530085037.543042&period=86400
    Cached https://poloniex.com/public?command=returnChartData&currencyPair=USDT_ETH&start=1451577600.0&end=1530085037.543042&period=86400 at USDT_ETH
    Downloading https://poloniex.com/public?command=returnChartData&currencyPair=USDT_LTC&start=1451577600.0&end=1530085038.851773&period=86400
    Cached https://poloniex.com/public?command=returnChartData&currencyPair=USDT_LTC&start=1451577600.0&end=1530085038.851773&period=86400 at USDT_LTC
    Downloading https://poloniex.com/public?command=returnChartData&currencyPair=USDT_XRP&start=1451577600.0&end=1530085039.734665&period=86400
    Cached https://poloniex.com/public?command=returnChartData&currencyPair=USDT_XRP&start=1451577600.0&end=1530085039.734665&period=86400 at USDT_XRP
    Downloading https://poloniex.com/public?command=returnChartData&currencyPair=USDT_XMR&start=1451577600.0&end=1530085041.185589&period=86400
    Cached https://poloniex.com/public?command=returnChartData&currencyPair=USDT_XMR&start=1451577600.0&end=1530085041.185589&period=86400 at USDT_XMR



```python
# merge price dataseries' into a single dataframe
altcoin_usd_datasets = merge_dfs_on_column(list(altcoin_data.values()), list(altcoin_data.keys()), 'weightedAverage')
# Remove "0" values 
altcoin_usd_datasets.replace(0, np.nan, inplace=True)
```


```python
altcoin_usd_datasets
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ETH</th>
      <th>LTC</th>
      <th>XMR</th>
      <th>XRP</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2016-01-01</th>
      <td>0.918600</td>
      <td>3.563049</td>
      <td>0.510780</td>
      <td>0.005999</td>
    </tr>
    <tr>
      <th>2016-01-02</th>
      <td>0.957000</td>
      <td>3.568911</td>
      <td>0.557775</td>
      <td>0.006127</td>
    </tr>
    <tr>
      <th>2016-01-03</th>
      <td>0.966507</td>
      <td>3.435222</td>
      <td>0.485490</td>
      <td>0.006149</td>
    </tr>
    <tr>
      <th>2016-01-04</th>
      <td>0.952906</td>
      <td>3.435230</td>
      <td>0.513003</td>
      <td>0.006149</td>
    </tr>
    <tr>
      <th>2016-01-05</th>
      <td>0.939680</td>
      <td>3.550613</td>
      <td>0.499595</td>
      <td>0.006149</td>
    </tr>
    <tr>
      <th>2016-01-06</th>
      <td>0.938506</td>
      <td>3.455243</td>
      <td>0.506093</td>
      <td>0.006104</td>
    </tr>
    <tr>
      <th>2016-01-07</th>
      <td>0.962401</td>
      <td>3.492098</td>
      <td>0.470000</td>
      <td>0.005969</td>
    </tr>
    <tr>
      <th>2016-01-08</th>
      <td>0.985378</td>
      <td>3.570236</td>
      <td>0.510517</td>
      <td>0.005886</td>
    </tr>
    <tr>
      <th>2016-01-09</th>
      <td>0.995999</td>
      <td>3.585341</td>
      <td>0.470000</td>
      <td>0.006137</td>
    </tr>
    <tr>
      <th>2016-01-10</th>
      <td>0.991087</td>
      <td>3.491326</td>
      <td>0.471312</td>
      <td>0.006137</td>
    </tr>
    <tr>
      <th>2016-01-11</th>
      <td>1.030464</td>
      <td>3.608803</td>
      <td>0.471000</td>
      <td>0.005701</td>
    </tr>
    <tr>
      <th>2016-01-12</th>
      <td>1.184094</td>
      <td>3.523703</td>
      <td>0.471000</td>
      <td>0.005701</td>
    </tr>
    <tr>
      <th>2016-01-13</th>
      <td>1.090158</td>
      <td>3.492231</td>
      <td>0.470297</td>
      <td>0.005687</td>
    </tr>
    <tr>
      <th>2016-01-14</th>
      <td>1.140453</td>
      <td>3.480432</td>
      <td>0.457853</td>
      <td>0.005572</td>
    </tr>
    <tr>
      <th>2016-01-15</th>
      <td>1.273829</td>
      <td>3.384859</td>
      <td>0.447766</td>
      <td>0.005319</td>
    </tr>
    <tr>
      <th>2016-01-16</th>
      <td>1.099764</td>
      <td>2.995942</td>
      <td>0.425839</td>
      <td>0.004900</td>
    </tr>
    <tr>
      <th>2016-01-17</th>
      <td>1.254971</td>
      <td>3.019984</td>
      <td>0.416666</td>
      <td>0.004897</td>
    </tr>
    <tr>
      <th>2016-01-18</th>
      <td>1.412402</td>
      <td>3.058518</td>
      <td>0.499993</td>
      <td>0.005408</td>
    </tr>
    <tr>
      <th>2016-01-19</th>
      <td>1.361609</td>
      <td>3.076332</td>
      <td>0.527162</td>
      <td>0.005258</td>
    </tr>
    <tr>
      <th>2016-01-20</th>
      <td>1.547963</td>
      <td>3.118459</td>
      <td>0.515656</td>
      <td>0.005245</td>
    </tr>
    <tr>
      <th>2016-01-21</th>
      <td>1.399045</td>
      <td>3.238037</td>
      <td>0.578312</td>
      <td>0.005566</td>
    </tr>
    <tr>
      <th>2016-01-22</th>
      <td>1.552112</td>
      <td>3.058768</td>
      <td>0.492848</td>
      <td>0.005201</td>
    </tr>
    <tr>
      <th>2016-01-23</th>
      <td>1.816847</td>
      <td>3.103798</td>
      <td>0.628181</td>
      <td>0.005141</td>
    </tr>
    <tr>
      <th>2016-01-24</th>
      <td>2.071194</td>
      <td>3.165229</td>
      <td>0.667244</td>
      <td>0.005122</td>
    </tr>
    <tr>
      <th>2016-01-25</th>
      <td>2.371175</td>
      <td>3.051658</td>
      <td>0.554825</td>
      <td>0.005122</td>
    </tr>
    <tr>
      <th>2016-01-26</th>
      <td>2.355269</td>
      <td>3.033605</td>
      <td>0.553421</td>
      <td>0.005412</td>
    </tr>
    <tr>
      <th>2016-01-27</th>
      <td>2.414311</td>
      <td>3.142906</td>
      <td>0.659182</td>
      <td>0.006385</td>
    </tr>
    <tr>
      <th>2016-01-28</th>
      <td>2.411107</td>
      <td>3.163685</td>
      <td>0.526709</td>
      <td>0.005804</td>
    </tr>
    <tr>
      <th>2016-01-29</th>
      <td>2.562321</td>
      <td>3.035938</td>
      <td>0.493655</td>
      <td>0.006630</td>
    </tr>
    <tr>
      <th>2016-01-30</th>
      <td>2.512182</td>
      <td>3.038959</td>
      <td>0.504056</td>
      <td>0.007635</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2018-05-29</th>
      <td>545.737092</td>
      <td>117.862281</td>
      <td>153.190995</td>
      <td>0.584856</td>
    </tr>
    <tr>
      <th>2018-05-30</th>
      <td>559.363763</td>
      <td>118.384026</td>
      <td>156.949726</td>
      <td>0.605588</td>
    </tr>
    <tr>
      <th>2018-05-31</th>
      <td>572.990210</td>
      <td>119.085306</td>
      <td>156.931970</td>
      <td>0.610469</td>
    </tr>
    <tr>
      <th>2018-06-01</th>
      <td>575.426882</td>
      <td>118.599275</td>
      <td>155.071680</td>
      <td>0.612187</td>
    </tr>
    <tr>
      <th>2018-06-02</th>
      <td>590.086335</td>
      <td>122.147996</td>
      <td>163.586298</td>
      <td>0.635585</td>
    </tr>
    <tr>
      <th>2018-06-03</th>
      <td>615.525843</td>
      <td>126.052890</td>
      <td>170.539870</td>
      <td>0.660091</td>
    </tr>
    <tr>
      <th>2018-06-04</th>
      <td>599.556071</td>
      <td>121.275475</td>
      <td>162.406788</td>
      <td>0.664816</td>
    </tr>
    <tr>
      <th>2018-06-05</th>
      <td>595.276573</td>
      <td>119.644840</td>
      <td>165.832882</td>
      <td>0.660014</td>
    </tr>
    <tr>
      <th>2018-06-06</th>
      <td>602.532216</td>
      <td>120.848434</td>
      <td>166.575553</td>
      <td>0.669818</td>
    </tr>
    <tr>
      <th>2018-06-07</th>
      <td>609.184375</td>
      <td>121.749574</td>
      <td>167.101285</td>
      <td>0.679829</td>
    </tr>
    <tr>
      <th>2018-06-08</th>
      <td>598.125166</td>
      <td>119.804857</td>
      <td>159.379850</td>
      <td>0.663900</td>
    </tr>
    <tr>
      <th>2018-06-09</th>
      <td>602.390825</td>
      <td>119.192960</td>
      <td>157.316172</td>
      <td>0.670576</td>
    </tr>
    <tr>
      <th>2018-06-10</th>
      <td>546.042944</td>
      <td>109.462657</td>
      <td>141.069571</td>
      <td>0.600596</td>
    </tr>
    <tr>
      <th>2018-06-11</th>
      <td>524.410084</td>
      <td>105.078832</td>
      <td>136.527187</td>
      <td>0.580023</td>
    </tr>
    <tr>
      <th>2018-06-12</th>
      <td>505.968076</td>
      <td>102.973435</td>
      <td>125.561917</td>
      <td>0.574205</td>
    </tr>
    <tr>
      <th>2018-06-13</th>
      <td>471.117286</td>
      <td>93.804958</td>
      <td>118.854273</td>
      <td>0.525484</td>
    </tr>
    <tr>
      <th>2018-06-14</th>
      <td>499.091862</td>
      <td>97.990790</td>
      <td>129.490168</td>
      <td>0.547208</td>
    </tr>
    <tr>
      <th>2018-06-15</th>
      <td>498.988329</td>
      <td>97.537612</td>
      <td>127.889559</td>
      <td>0.543665</td>
    </tr>
    <tr>
      <th>2018-06-16</th>
      <td>493.150314</td>
      <td>95.636537</td>
      <td>123.757971</td>
      <td>0.530265</td>
    </tr>
    <tr>
      <th>2018-06-17</th>
      <td>502.197091</td>
      <td>96.830339</td>
      <td>124.467749</td>
      <td>0.530279</td>
    </tr>
    <tr>
      <th>2018-06-18</th>
      <td>505.032334</td>
      <td>96.342016</td>
      <td>126.447257</td>
      <td>0.532905</td>
    </tr>
    <tr>
      <th>2018-06-19</th>
      <td>528.777945</td>
      <td>98.789820</td>
      <td>127.423466</td>
      <td>0.546285</td>
    </tr>
    <tr>
      <th>2018-06-20</th>
      <td>529.218135</td>
      <td>97.167110</td>
      <td>122.046611</td>
      <td>0.534537</td>
    </tr>
    <tr>
      <th>2018-06-21</th>
      <td>531.358767</td>
      <td>97.080116</td>
      <td>123.249020</td>
      <td>0.536327</td>
    </tr>
    <tr>
      <th>2018-06-22</th>
      <td>483.728410</td>
      <td>87.786004</td>
      <td>109.892480</td>
      <td>0.497325</td>
    </tr>
    <tr>
      <th>2018-06-23</th>
      <td>468.989007</td>
      <td>83.348104</td>
      <td>115.987754</td>
      <td>0.485186</td>
    </tr>
    <tr>
      <th>2018-06-24</th>
      <td>442.268233</td>
      <td>78.543286</td>
      <td>113.958179</td>
      <td>0.463577</td>
    </tr>
    <tr>
      <th>2018-06-25</th>
      <td>459.961039</td>
      <td>81.598908</td>
      <td>127.035981</td>
      <td>0.481141</td>
    </tr>
    <tr>
      <th>2018-06-26</th>
      <td>445.131520</td>
      <td>80.154925</td>
      <td>124.295410</td>
      <td>0.474318</td>
    </tr>
    <tr>
      <th>2018-06-27</th>
      <td>429.838027</td>
      <td>75.881479</td>
      <td>124.458243</td>
      <td>0.456312</td>
    </tr>
  </tbody>
</table>
<p>909 rows × 4 columns</p>
</div>



## 資料整合：加入其它貨幣合併
#### 合併上面抓取到的比特幣報價與其他虛擬貨幣的報價，整合為單一的DataFrame做為資料集


```python
hist = pd.merge(btc_usd_datasets,altcoin_usd_datasets, left_index=True, right_index=True)
```

## 將14天的價格變化資料分為訓練及測試集
#### 利用前面建立的函數快速輕鬆地建立好訓練與測試用的資料集


```python
train, test, X_train, X_test, y_train, y_test = prepare_data(hist)
```

## 訓練LSTM 模型


```python
model = build_lstm_model(X_train, output_size=1)
history = model.fit(X_train, y_train, epochs=100, batch_size=4)
```

    Epoch 1/100
    804/804 [==============================] - 4s 5ms/step - loss: 0.0833
    Epoch 2/100
    804/804 [==============================] - 3s 4ms/step - loss: 0.0544
    Epoch 3/100
    804/804 [==============================] - 3s 4ms/step - loss: 0.0468
    Epoch 4/100
    804/804 [==============================] - 3s 4ms/step - loss: 0.0461
    Epoch 5/100
    804/804 [==============================] - 3s 4ms/step - loss: 0.0422
    Epoch 6/100
    804/804 [==============================] - 3s 4ms/step - loss: 0.0399
    Epoch 7/100
    804/804 [==============================] - 4s 5ms/step - loss: 0.0397
    Epoch 8/100
    804/804 [==============================] - 3s 4ms/step - loss: 0.0359
    Epoch 9/100
    804/804 [==============================] - 4s 4ms/step - loss: 0.0365
    Epoch 10/100
    804/804 [==============================] - 3s 4ms/step - loss: 0.0366
    Epoch 11/100
    804/804 [==============================] - 3s 4ms/step - loss: 0.0354
    Epoch 12/100
    804/804 [==============================] - 3s 4ms/step - loss: 0.0357
    Epoch 13/100
    804/804 [==============================] - 3s 4ms/step - loss: 0.0344
    Epoch 14/100
    804/804 [==============================] - 3s 4ms/step - loss: 0.0342
    Epoch 15/100
    804/804 [==============================] - 3s 4ms/step - loss: 0.0330
    Epoch 16/100
    804/804 [==============================] - 3s 4ms/step - loss: 0.0330
    Epoch 17/100
    804/804 [==============================] - 3s 4ms/step - loss: 0.0324
    Epoch 18/100
    804/804 [==============================] - 3s 4ms/step - loss: 0.0330
    Epoch 19/100
    804/804 [==============================] - 3s 4ms/step - loss: 0.0314
    Epoch 20/100
    804/804 [==============================] - 4s 5ms/step - loss: 0.0314
    Epoch 21/100
    804/804 [==============================] - 3s 4ms/step - loss: 0.0332
    Epoch 22/100
    804/804 [==============================] - 3s 4ms/step - loss: 0.0318
    Epoch 23/100
    804/804 [==============================] - 4s 4ms/step - loss: 0.0314
    Epoch 24/100
    804/804 [==============================] - 3s 4ms/step - loss: 0.0296
    Epoch 25/100
    804/804 [==============================] - 4s 4ms/step - loss: 0.0308
    Epoch 26/100
    804/804 [==============================] - 3s 4ms/step - loss: 0.0295
    Epoch 27/100
    804/804 [==============================] - 3s 4ms/step - loss: 0.0300
    Epoch 28/100
    804/804 [==============================] - 4s 4ms/step - loss: 0.0314
    Epoch 29/100
    804/804 [==============================] - 3s 4ms/step - loss: 0.0296
    Epoch 30/100
    804/804 [==============================] - 3s 4ms/step - loss: 0.0309
    Epoch 31/100
    804/804 [==============================] - 4s 5ms/step - loss: 0.0295
    Epoch 32/100
    804/804 [==============================] - 3s 4ms/step - loss: 0.0309
    Epoch 33/100
    804/804 [==============================] - 4s 4ms/step - loss: 0.0308
    Epoch 34/100
    804/804 [==============================] - 4s 4ms/step - loss: 0.0300
    Epoch 35/100
    804/804 [==============================] - 4s 4ms/step - loss: 0.0300
    Epoch 36/100
    804/804 [==============================] - 4s 4ms/step - loss: 0.0301
    Epoch 37/100
    804/804 [==============================] - 4s 4ms/step - loss: 0.0289
    Epoch 38/100
    804/804 [==============================] - 4s 5ms/step - loss: 0.0291
    Epoch 39/100
    804/804 [==============================] - 3s 4ms/step - loss: 0.0294
    Epoch 40/100
    804/804 [==============================] - 3s 4ms/step - loss: 0.0294
    Epoch 41/100
    804/804 [==============================] - 3s 4ms/step - loss: 0.0302
    Epoch 42/100
    804/804 [==============================] - 3s 4ms/step - loss: 0.0294
    Epoch 43/100
    804/804 [==============================] - 3s 4ms/step - loss: 0.0299
    Epoch 44/100
    804/804 [==============================] - 3s 4ms/step - loss: 0.0310
    Epoch 45/100
    804/804 [==============================] - 4s 4ms/step - loss: 0.0315
    Epoch 46/100
    804/804 [==============================] - 3s 4ms/step - loss: 0.0300
    Epoch 47/100
    804/804 [==============================] - 3s 4ms/step - loss: 0.0290
    Epoch 48/100
    804/804 [==============================] - 3s 4ms/step - loss: 0.0285
    Epoch 49/100
    804/804 [==============================] - 3s 4ms/step - loss: 0.0289
    Epoch 50/100
    804/804 [==============================] - 3s 4ms/step - loss: 0.0301
    Epoch 51/100
    804/804 [==============================] - 3s 4ms/step - loss: 0.0300
    Epoch 52/100
    804/804 [==============================] - 3s 4ms/step - loss: 0.0305
    Epoch 53/100
    804/804 [==============================] - 3s 4ms/step - loss: 0.0296
    Epoch 54/100
    804/804 [==============================] - 3s 4ms/step - loss: 0.0286
    Epoch 55/100
    804/804 [==============================] - 3s 4ms/step - loss: 0.0276
    Epoch 56/100
    804/804 [==============================] - 4s 4ms/step - loss: 0.0292
    Epoch 57/100
    804/804 [==============================] - 4s 5ms/step - loss: 0.0282
    Epoch 58/100
    804/804 [==============================] - 4s 5ms/step - loss: 0.0302
    Epoch 59/100
    804/804 [==============================] - 3s 4ms/step - loss: 0.0305
    Epoch 60/100
    804/804 [==============================] - 3s 4ms/step - loss: 0.0292
    Epoch 61/100
    804/804 [==============================] - 4s 4ms/step - loss: 0.0291
    Epoch 62/100
    804/804 [==============================] - 3s 4ms/step - loss: 0.0278
    Epoch 63/100
    804/804 [==============================] - 3s 4ms/step - loss: 0.0293
    Epoch 64/100
    804/804 [==============================] - 3s 4ms/step - loss: 0.0300
    Epoch 65/100
    804/804 [==============================] - 3s 4ms/step - loss: 0.0293
    Epoch 66/100
    804/804 [==============================] - 3s 4ms/step - loss: 0.0297
    Epoch 67/100
    804/804 [==============================] - 3s 4ms/step - loss: 0.0288
    Epoch 68/100
    804/804 [==============================] - 3s 4ms/step - loss: 0.0302
    Epoch 69/100
    804/804 [==============================] - 3s 4ms/step - loss: 0.0287
    Epoch 70/100
    804/804 [==============================] - 3s 4ms/step - loss: 0.0296
    Epoch 71/100
    804/804 [==============================] - 3s 4ms/step - loss: 0.0298
    Epoch 72/100
    804/804 [==============================] - 3s 4ms/step - loss: 0.0295
    Epoch 73/100
    804/804 [==============================] - 3s 4ms/step - loss: 0.0300
    Epoch 74/100
    804/804 [==============================] - 3s 4ms/step - loss: 0.0302
    Epoch 75/100
    804/804 [==============================] - 3s 4ms/step - loss: 0.0292
    Epoch 76/100
    804/804 [==============================] - 4s 5ms/step - loss: 0.0290
    Epoch 77/100
    804/804 [==============================] - 3s 4ms/step - loss: 0.0295
    Epoch 78/100
    804/804 [==============================] - 3s 4ms/step - loss: 0.0285
    Epoch 79/100
    804/804 [==============================] - 4s 5ms/step - loss: 0.0291
    Epoch 80/100
    804/804 [==============================] - 3s 4ms/step - loss: 0.0291
    Epoch 81/100
    804/804 [==============================] - 4s 4ms/step - loss: 0.0288
    Epoch 82/100
    804/804 [==============================] - 3s 4ms/step - loss: 0.0291
    Epoch 83/100
    804/804 [==============================] - 3s 4ms/step - loss: 0.0301
    Epoch 84/100
    804/804 [==============================] - 3s 4ms/step - loss: 0.0289
    Epoch 85/100
    804/804 [==============================] - 3s 4ms/step - loss: 0.0283
    Epoch 86/100
    804/804 [==============================] - 3s 4ms/step - loss: 0.0285
    Epoch 87/100
    804/804 [==============================] - 3s 4ms/step - loss: 0.0302
    Epoch 88/100
    804/804 [==============================] - 3s 4ms/step - loss: 0.0285
    Epoch 89/100
    804/804 [==============================] - 3s 4ms/step - loss: 0.0295
    Epoch 90/100
    804/804 [==============================] - 3s 4ms/step - loss: 0.0292
    Epoch 91/100
    804/804 [==============================] - 3s 4ms/step - loss: 0.0294
    Epoch 92/100
    804/804 [==============================] - 3s 4ms/step - loss: 0.0293
    Epoch 93/100
    804/804 [==============================] - 3s 4ms/step - loss: 0.0303
    Epoch 94/100
    804/804 [==============================] - 3s 4ms/step - loss: 0.0296
    Epoch 95/100
    804/804 [==============================] - 3s 4ms/step - loss: 0.0290
    Epoch 96/100
    804/804 [==============================] - 3s 4ms/step - loss: 0.0293
    Epoch 97/100
    804/804 [==============================] - 3s 4ms/step - loss: 0.0291
    Epoch 98/100
    804/804 [==============================] - 3s 4ms/step - loss: 0.0288
    Epoch 99/100
    804/804 [==============================] - 3s 4ms/step - loss: 0.0289
    Epoch 100/100
    804/804 [==============================] - 3s 4ms/step - loss: 0.0274


## 還原結果
#### 因為資料集都是經過normailise的數值
#### 所以這邊需要再把normailise的數值還原成實際的價格


```python
target_col='average'
window=14
targets = test[target_col][window:]
preds = model.predict(X_test).squeeze()
preds = test.average.values[:-window] * (preds + 1)
preds = pd.Series(index=targets.index, data=preds)
```

## 繪製30天比較圖


```python
n = 30
line_plot(targets[-n:], preds[-n:], 'actual', 'prediction')
```


![png](output_44_0.png)


