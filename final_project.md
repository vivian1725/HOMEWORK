
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
from keras.models import load_model
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import Activation
from scipy import stats
import statsmodels.api as sm
import warnings
from itertools import product
from datetime import datetime
warnings.filterwarnings('ignore')
plt.style.use('seaborn-poster')
```

    Using TensorFlow backend.


# 定義函數

### 下載來自Quandl的 Bitcoin 資料集
### Quandl本身提供了Python的資料擷取套件 ，所以只需要簡單的設定就可以輕鬆地獲得資料
### 中間利用pickle作資料的備份
### api_key可以不用填沒問題，但會有連接上線 ，所以還是建議到Quandl申請免費的帳號獲取API KEY


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
#### 藉由Poloniex提供的Web service來抓去幣價的JSON檔 ，所以先建立一個函式用抓取JSON


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
#### 實際利用上面的函式來抓去JSON檔再利用pandas儲存成DataFrame的格式方便後續的操作


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
#### 把資料其切分成訓練集與測試集 ，這邊的設定是抓90%數據訓練，10%數據測試


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
### 因為後續不是只有BTC的價錢，還有加入其他的虛擬貨幣一起 
### 所以為了避免受不同幣種幣價原本的高低影響，所以做normailise
#### 那這邊的做法是讓價錢變成跟第一天的價錢的漲跌幅
#### 也就是說假設第一天價錢是8000，第二天價錢變成10000，第三天價錢又變回8000
#### 則在normailise之後就會變成0、0.25、0 ，那14天的資料也都是以照這樣的邏輯進行


```python
def normalise_zero_base(df):
    """ Normalise dataframe column-wise to reflect changes with
        respect to first entry.
    """
    return df / df.iloc[0] - 1
```

## 建立函數快速LSTM建模所需的資料模式
### 預設是14天週期，並使用normailise
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
#### Quandl上有持續在更新且以美金報價的交易所為下列4所
#### 因為虛擬貨幣的交易在各國的規範日益嚴苛 , 所以能使用法幣出入金的交易所不多


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


### 僅捉取2016年之後的加權價格 , 並增加一欄平均值作為訓練的目標價格
### 只使用2016年之後的價格主要是為了配合下面引進其他虛擬貨幣的部分
### 捉取的資料期有部分缺失或為0的資料 , 缺失或為0的資料則利用當天其他交易所的平均價格替補


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
exchange_Volume = ['COINBASE-V', 'BITSTAMP-V', 'ITBIT-V', 'KRAKEN-V']
```


```python
btc_usd_datasets_Volume = merge_dfs_on_column(list(exchange_data.values()), exchange_Volume , 'Volume (BTC)')
btc_usd_datasets_Volume = btc_usd_datasets_Volume.loc[btc_usd_datasets_Volume.index >= '2016-01-01']
btc_usd_datasets_Volume['sum_V'] = btc_usd_datasets_Volume.sum(axis=1)
btc_usd_datasets_Volume =pd.DataFrame(btc_usd_datasets_Volume['sum_V'])
btc_usd_datasets=pd.merge(btc_usd_datasets,btc_usd_datasets_Volume , left_index=True, right_index=True)
btc_usd_datasets_Volume=normalise_zero_base(btc_usd_datasets)
```


```python
btc_usd_datasets
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
      <th>BITSTAMP</th>
      <th>COINBASE</th>
      <th>ITBIT</th>
      <th>KRAKEN</th>
      <th>average</th>
      <th>sum_V</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2016-01-01</th>
      <td>433.086003</td>
      <td>433.358101</td>
      <td>431.854189</td>
      <td>433.197419</td>
      <td>432.873928</td>
      <td>8749.673805</td>
    </tr>
    <tr>
      <th>2016-01-02</th>
      <td>433.292697</td>
      <td>435.286346</td>
      <td>433.211856</td>
      <td>432.989873</td>
      <td>433.695193</td>
      <td>6591.310053</td>
    </tr>
    <tr>
      <th>2016-01-03</th>
      <td>428.595713</td>
      <td>430.697845</td>
      <td>429.316383</td>
      <td>427.434460</td>
      <td>429.011100</td>
      <td>9678.402186</td>
    </tr>
    <tr>
      <th>2016-01-04</th>
      <td>432.834487</td>
      <td>433.938214</td>
      <td>432.178487</td>
      <td>430.631979</td>
      <td>432.395792</td>
      <td>13463.971066</td>
    </tr>
    <tr>
      <th>2016-01-05</th>
      <td>432.053592</td>
      <td>433.300199</td>
      <td>432.729037</td>
      <td>430.513601</td>
      <td>432.149107</td>
      <td>10381.184703</td>
    </tr>
    <tr>
      <th>2016-01-06</th>
      <td>430.138415</td>
      <td>431.829006</td>
      <td>430.287749</td>
      <td>428.381249</td>
      <td>430.159105</td>
      <td>12809.357391</td>
    </tr>
    <tr>
      <th>2016-01-07</th>
      <td>447.705438</td>
      <td>450.860143</td>
      <td>446.215816</td>
      <td>446.620606</td>
      <td>447.850501</td>
      <td>34986.550228</td>
    </tr>
    <tr>
      <th>2016-01-08</th>
      <td>453.746023</td>
      <td>456.297446</td>
      <td>455.519939</td>
      <td>454.347885</td>
      <td>454.977823</td>
      <td>21181.179401</td>
    </tr>
    <tr>
      <th>2016-01-09</th>
      <td>449.706449</td>
      <td>452.112265</td>
      <td>450.358646</td>
      <td>454.445267</td>
      <td>451.655657</td>
      <td>9862.746028</td>
    </tr>
    <tr>
      <th>2016-01-10</th>
      <td>444.613427</td>
      <td>446.888543</td>
      <td>444.230268</td>
      <td>447.733606</td>
      <td>445.866461</td>
      <td>9171.058738</td>
    </tr>
    <tr>
      <th>2016-01-11</th>
      <td>445.397624</td>
      <td>449.175560</td>
      <td>446.491196</td>
      <td>449.573412</td>
      <td>447.659448</td>
      <td>15308.559910</td>
    </tr>
    <tr>
      <th>2016-01-12</th>
      <td>442.635775</td>
      <td>445.906991</td>
      <td>444.171883</td>
      <td>448.741337</td>
      <td>445.363997</td>
      <td>17392.029789</td>
    </tr>
    <tr>
      <th>2016-01-13</th>
      <td>429.865880</td>
      <td>431.182928</td>
      <td>429.705328</td>
      <td>434.424323</td>
      <td>431.294615</td>
      <td>22499.348192</td>
    </tr>
    <tr>
      <th>2016-01-14</th>
      <td>430.595248</td>
      <td>431.932933</td>
      <td>431.179699</td>
      <td>432.608325</td>
      <td>431.579051</td>
      <td>11583.042117</td>
    </tr>
    <tr>
      <th>2016-01-15</th>
      <td>395.327840</td>
      <td>396.047745</td>
      <td>397.139540</td>
      <td>397.019737</td>
      <td>396.383715</td>
      <td>82756.380098</td>
    </tr>
    <tr>
      <th>2016-01-16</th>
      <td>372.273967</td>
      <td>371.902981</td>
      <td>369.646494</td>
      <td>375.253691</td>
      <td>372.269283</td>
      <td>47563.958484</td>
    </tr>
    <tr>
      <th>2016-01-17</th>
      <td>382.814307</td>
      <td>384.891976</td>
      <td>383.395818</td>
      <td>388.136221</td>
      <td>384.809581</td>
      <td>18566.791959</td>
    </tr>
    <tr>
      <th>2016-01-18</th>
      <td>382.049880</td>
      <td>380.191326</td>
      <td>382.612068</td>
      <td>386.205222</td>
      <td>382.764624</td>
      <td>17931.682925</td>
    </tr>
    <tr>
      <th>2016-01-19</th>
      <td>381.875751</td>
      <td>379.181472</td>
      <td>383.059396</td>
      <td>384.537452</td>
      <td>382.163518</td>
      <td>18685.000360</td>
    </tr>
    <tr>
      <th>2016-01-20</th>
      <td>400.098151</td>
      <td>398.347144</td>
      <td>402.148102</td>
      <td>402.508822</td>
      <td>400.775555</td>
      <td>47943.918698</td>
    </tr>
    <tr>
      <th>2016-01-21</th>
      <td>412.828309</td>
      <td>412.743387</td>
      <td>412.505932</td>
      <td>415.302977</td>
      <td>413.345151</td>
      <td>22129.636515</td>
    </tr>
    <tr>
      <th>2016-01-22</th>
      <td>389.807055</td>
      <td>392.281630</td>
      <td>389.510420</td>
      <td>394.484037</td>
      <td>391.520786</td>
      <td>34867.230657</td>
    </tr>
    <tr>
      <th>2016-01-23</th>
      <td>388.048747</td>
      <td>390.415930</td>
      <td>387.328244</td>
      <td>389.082836</td>
      <td>388.718939</td>
      <td>15856.569111</td>
    </tr>
    <tr>
      <th>2016-01-24</th>
      <td>399.438070</td>
      <td>400.545291</td>
      <td>399.340155</td>
      <td>395.583534</td>
      <td>398.726762</td>
      <td>11898.993957</td>
    </tr>
    <tr>
      <th>2016-01-25</th>
      <td>394.123921</td>
      <td>394.071226</td>
      <td>395.813026</td>
      <td>395.971571</td>
      <td>394.994936</td>
      <td>18312.790925</td>
    </tr>
    <tr>
      <th>2016-01-26</th>
      <td>392.462031</td>
      <td>392.911756</td>
      <td>393.070622</td>
      <td>396.440152</td>
      <td>393.721141</td>
      <td>17369.342003</td>
    </tr>
    <tr>
      <th>2016-01-27</th>
      <td>394.335429</td>
      <td>394.299920</td>
      <td>394.663989</td>
      <td>394.433113</td>
      <td>394.433113</td>
      <td>12045.962727</td>
    </tr>
    <tr>
      <th>2016-01-28</th>
      <td>382.112568</td>
      <td>384.168262</td>
      <td>383.986475</td>
      <td>383.422435</td>
      <td>383.422435</td>
      <td>23693.307497</td>
    </tr>
    <tr>
      <th>2016-01-29</th>
      <td>375.405357</td>
      <td>375.802755</td>
      <td>375.123574</td>
      <td>375.443895</td>
      <td>375.443895</td>
      <td>31228.268957</td>
    </tr>
    <tr>
      <th>2016-01-30</th>
      <td>377.095168</td>
      <td>378.898120</td>
      <td>378.402363</td>
      <td>378.131883</td>
      <td>378.131883</td>
      <td>7524.346396</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2018-05-29</th>
      <td>7313.561835</td>
      <td>7319.555427</td>
      <td>7256.355273</td>
      <td>7320.208337</td>
      <td>7302.420218</td>
      <td>32912.690049</td>
    </tr>
    <tr>
      <th>2018-05-30</th>
      <td>7433.678883</td>
      <td>7410.092220</td>
      <td>7435.904061</td>
      <td>7421.249082</td>
      <td>7425.231062</td>
      <td>23642.821280</td>
    </tr>
    <tr>
      <th>2018-05-31</th>
      <td>7513.858285</td>
      <td>7509.528862</td>
      <td>7507.800687</td>
      <td>7503.460232</td>
      <td>7508.662016</td>
      <td>20317.242457</td>
    </tr>
    <tr>
      <th>2018-06-01</th>
      <td>7480.517612</td>
      <td>7461.489319</td>
      <td>7476.332680</td>
      <td>7468.174673</td>
      <td>7471.628571</td>
      <td>20707.480359</td>
    </tr>
    <tr>
      <th>2018-06-02</th>
      <td>7595.743627</td>
      <td>7602.497070</td>
      <td>7601.029234</td>
      <td>7611.456144</td>
      <td>7602.681519</td>
      <td>13762.744629</td>
    </tr>
    <tr>
      <th>2018-06-03</th>
      <td>7691.812400</td>
      <td>7695.990426</td>
      <td>7706.696376</td>
      <td>7699.780629</td>
      <td>7698.569958</td>
      <td>13923.427246</td>
    </tr>
    <tr>
      <th>2018-06-04</th>
      <td>7563.667686</td>
      <td>7546.190384</td>
      <td>7572.952961</td>
      <td>7559.445365</td>
      <td>7560.564099</td>
      <td>20679.287852</td>
    </tr>
    <tr>
      <th>2018-06-05</th>
      <td>7494.371636</td>
      <td>7507.619130</td>
      <td>7487.076475</td>
      <td>7498.873124</td>
      <td>7496.985091</td>
      <td>23388.724532</td>
    </tr>
    <tr>
      <th>2018-06-06</th>
      <td>7605.202879</td>
      <td>7598.733666</td>
      <td>7600.973325</td>
      <td>7588.131850</td>
      <td>7598.260430</td>
      <td>20012.225930</td>
    </tr>
    <tr>
      <th>2018-06-07</th>
      <td>7695.606549</td>
      <td>7690.472766</td>
      <td>7696.137207</td>
      <td>7690.226731</td>
      <td>7693.110813</td>
      <td>18364.752663</td>
    </tr>
    <tr>
      <th>2018-06-08</th>
      <td>7617.315370</td>
      <td>7621.473462</td>
      <td>7614.254115</td>
      <td>7622.616019</td>
      <td>7618.914741</td>
      <td>12775.157679</td>
    </tr>
    <tr>
      <th>2018-06-09</th>
      <td>7598.005433</td>
      <td>7599.669124</td>
      <td>7590.016189</td>
      <td>7588.716522</td>
      <td>7594.101817</td>
      <td>8126.279316</td>
    </tr>
    <tr>
      <th>2018-06-10</th>
      <td>6965.698109</td>
      <td>6942.736380</td>
      <td>6966.179545</td>
      <td>6986.947837</td>
      <td>6965.390468</td>
      <td>48541.296127</td>
    </tr>
    <tr>
      <th>2018-06-11</th>
      <td>6754.939110</td>
      <td>6761.736699</td>
      <td>6747.543538</td>
      <td>6761.810263</td>
      <td>6756.507403</td>
      <td>31992.052361</td>
    </tr>
    <tr>
      <th>2018-06-12</th>
      <td>6694.118400</td>
      <td>6659.272110</td>
      <td>6667.092383</td>
      <td>6679.180626</td>
      <td>6674.915880</td>
      <td>29768.361845</td>
    </tr>
    <tr>
      <th>2018-06-13</th>
      <td>6398.309520</td>
      <td>6371.255598</td>
      <td>6400.374293</td>
      <td>6358.993208</td>
      <td>6382.233155</td>
      <td>52740.392038</td>
    </tr>
    <tr>
      <th>2018-06-14</th>
      <td>6493.931518</td>
      <td>6503.001341</td>
      <td>6481.469984</td>
      <td>6501.680531</td>
      <td>6495.020844</td>
      <td>42735.730162</td>
    </tr>
    <tr>
      <th>2018-06-15</th>
      <td>6545.672008</td>
      <td>6535.241459</td>
      <td>6548.712561</td>
      <td>6530.954478</td>
      <td>6540.145126</td>
      <td>22226.273840</td>
    </tr>
    <tr>
      <th>2018-06-16</th>
      <td>6460.211171</td>
      <td>6455.774489</td>
      <td>6455.335062</td>
      <td>6451.717180</td>
      <td>6455.759475</td>
      <td>12201.679625</td>
    </tr>
    <tr>
      <th>2018-06-17</th>
      <td>6509.017847</td>
      <td>6498.813185</td>
      <td>6498.136062</td>
      <td>6499.854811</td>
      <td>6501.455476</td>
      <td>9176.816347</td>
    </tr>
    <tr>
      <th>2018-06-18</th>
      <td>6551.333545</td>
      <td>6586.614178</td>
      <td>6580.586341</td>
      <td>6575.871276</td>
      <td>6573.601335</td>
      <td>19334.122130</td>
    </tr>
    <tr>
      <th>2018-06-19</th>
      <td>6731.628819</td>
      <td>6728.600883</td>
      <td>6726.481215</td>
      <td>6730.366504</td>
      <td>6729.269355</td>
      <td>16865.428771</td>
    </tr>
    <tr>
      <th>2018-06-20</th>
      <td>6679.719281</td>
      <td>6692.957851</td>
      <td>6681.077994</td>
      <td>6679.288215</td>
      <td>6683.260835</td>
      <td>17914.311412</td>
    </tr>
    <tr>
      <th>2018-06-21</th>
      <td>6731.905067</td>
      <td>6731.627631</td>
      <td>6730.904546</td>
      <td>6729.491059</td>
      <td>6730.982076</td>
      <td>13265.411911</td>
    </tr>
    <tr>
      <th>2018-06-22</th>
      <td>6283.202163</td>
      <td>6241.882496</td>
      <td>6287.060492</td>
      <td>6253.716124</td>
      <td>6266.465319</td>
      <td>52492.267032</td>
    </tr>
    <tr>
      <th>2018-06-23</th>
      <td>6137.488639</td>
      <td>6133.479208</td>
      <td>6121.887751</td>
      <td>6133.505460</td>
      <td>6131.590265</td>
      <td>17195.259088</td>
    </tr>
    <tr>
      <th>2018-06-24</th>
      <td>5987.551809</td>
      <td>6020.179715</td>
      <td>6008.756939</td>
      <td>5990.530329</td>
      <td>6001.754698</td>
      <td>39287.958124</td>
    </tr>
    <tr>
      <th>2018-06-25</th>
      <td>6204.806754</td>
      <td>6212.208982</td>
      <td>6197.104192</td>
      <td>6209.353804</td>
      <td>6205.868433</td>
      <td>28681.231501</td>
    </tr>
    <tr>
      <th>2018-06-26</th>
      <td>6186.030207</td>
      <td>6175.888527</td>
      <td>6184.709258</td>
      <td>6172.424748</td>
      <td>6179.763185</td>
      <td>22865.746868</td>
    </tr>
    <tr>
      <th>2018-06-27</th>
      <td>6088.770534</td>
      <td>6090.949848</td>
      <td>6087.267937</td>
      <td>6091.902635</td>
      <td>6089.722739</td>
      <td>21026.721350</td>
    </tr>
  </tbody>
</table>
<p>909 rows × 6 columns</p>
</div>




```python
line_plot(btc_usd_datasets_Volume.average,btc_usd_datasets_Volume.sum_V, 'avage', 'Volume')
```


![png](output_32_0.png)


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

    Downloading https://poloniex.com/public?command=returnChartData&currencyPair=USDT_ETH&start=1451577600.0&end=1530179875.045544&period=86400
    Cached https://poloniex.com/public?command=returnChartData&currencyPair=USDT_ETH&start=1451577600.0&end=1530179875.045544&period=86400 at USDT_ETH
    Downloading https://poloniex.com/public?command=returnChartData&currencyPair=USDT_LTC&start=1451577600.0&end=1530179876.887559&period=86400
    Cached https://poloniex.com/public?command=returnChartData&currencyPair=USDT_LTC&start=1451577600.0&end=1530179876.887559&period=86400 at USDT_LTC
    Downloading https://poloniex.com/public?command=returnChartData&currencyPair=USDT_XRP&start=1451577600.0&end=1530179877.934&period=86400
    Cached https://poloniex.com/public?command=returnChartData&currencyPair=USDT_XRP&start=1451577600.0&end=1530179877.934&period=86400 at USDT_XRP
    Downloading https://poloniex.com/public?command=returnChartData&currencyPair=USDT_XMR&start=1451577600.0&end=1530179879.295549&period=86400
    Cached https://poloniex.com/public?command=returnChartData&currencyPair=USDT_XMR&start=1451577600.0&end=1530179879.295549&period=86400 at USDT_XMR



```python
# merge price dataseries' into a single dataframe
altcoin_usd_datasets = merge_dfs_on_column(list(altcoin_data.values()), list(altcoin_data.keys()), 'weightedAverage')
# Remove "0" values 
altcoin_usd_datasets.replace(0, np.nan, inplace=True)
```


```python
# PLOTS
fig = plt.figure(figsize=[15, 7])
plt.suptitle('Cryptocurrency,  USD', fontsize=22)

plt.subplot(221)
plt.plot(altcoin_usd_datasets.ETH, '-', label='ETH')
plt.legend()

plt.subplot(222)
plt.plot(altcoin_usd_datasets.LTC, '-', label='LTC')
plt.legend()

plt.subplot(223)
plt.plot(altcoin_usd_datasets.XMR, '-', label='XMR')
plt.legend()

plt.subplot(224)
plt.plot(altcoin_usd_datasets.XRP, '-', label='XRP')
plt.legend()

# plt.tight_layout()
plt.show()
```


![png](output_36_0.png)


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
model = load_model('10萬epochs.octet-stream')
#model = build_lstm_model(X_train, output_size=1)
##history = model.fit(X_train, y_train, epochs=10, batch_size=4)
```

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


![png](output_46_0.png)


## 透過ARMIA模型預測Bitcoin比特幣未來1個月交易價格
## 資料檢索
#### 使用Kaggle提供Bitcoin2012-2018每分鐘CSV交易資料集
```python
df = pd.read_csv('../input/bitstampUSD_1-min_data_2012-01-01_to_2018-03-27.csv')
df.head()
```

Timestamp	Open	High	Low	Close	Volume_(BTC)	Volume_(Currency)	Weighted_Price
0	1325317920	4.39	4.39	4.39	4.39	0.455581	2.0	4.39
1	1325317980	4.39	4.39	4.39	4.39	0.455581	2.0	4.39
2	1325318040	4.39	4.39	4.39	4.39	0.455581	2.0	4.39
3	1325318100	4.39	4.39	4.39	4.39	0.455581	2.0	4.39
4	1325318160	4.39	4.39	4.39	4.39	0.455581	2.0	4.39

## 重新採樣每日、每月、每年、每季資料
```python
df.Timestamp = pd.to_datetime(df.Timestamp, unit='s')
df.index = df.Timestamp
df = df.resample('D').mean()
df_month = df.resample('M').mean()
df_year = df.resample('A-DEC').mean()
df_Q = df.resample('Q-DEC').mean()
```

## 顯示Bitcoin的美元走勢
```python
fig = plt.figure(figsize=[15, 7])
plt.suptitle('Bitcoin exchanges, mean USD', fontsize=22)

plt.subplot(221)
plt.plot(df.Weighted_Price, '-', label='By Days')
plt.legend()

plt.subplot(222)
plt.plot(df_month.Weighted_Price, '-', label='By Months')
plt.legend()

plt.subplot(223)
plt.plot(df_Q.Weighted_Price, '-', label='By Quarters')
plt.legend()

plt.subplot(224)
plt.plot(df_year.Weighted_Price, '-', label='By Years')
plt.legend()

plt.show()
```
![png](BitcoinUSD.png)


## 模型選擇
#### 透過自相關函數和偏自相關函數建立模型

```python
plt.figure(figsize=(15,7))
ax = plt.subplot(211)
sm.graphics.tsa.plot_acf(df_month.prices_box_diff2[13:].values.squeeze(), lags=48, ax=ax)
ax = plt.subplot(212)
sm.graphics.tsa.plot_pacf(df_month.prices_box_diff2[13:].values.squeeze(), lags=48, ax=ax)
plt.tight_layout()
plt.show()
```
![png](Autocorrelation.png)


#### 最透過反轉Box-Cox轉換函數後，預測出價格走勢

```python
def invboxcox(y,lmbda):
   if lmbda == 0:
      return(np.exp(y))
   else:
      return(np.exp(np.log(lmbda*y+1)/lmbda))
      
df_month2 = df_month[['Weighted_Price']]
date_list = [datetime(2017, 6, 30), datetime(2017, 7, 31), datetime(2017, 8, 31), datetime(2017, 9, 30), 
             datetime(2017, 10, 31), datetime(2017, 11, 30), datetime(2017, 12, 31), datetime(2018, 1, 31),datetime(2018, 5, 31),
             datetime(2018, 10, 30)]
future = pd.DataFrame(index=date_list, columns= df_month.columns)
df_month2 = pd.concat([df_month2, future])
df_month2['forecast'] = invboxcox(best_model.predict(start=0, end=75), lmbda)
plt.figure(figsize=(15,7))
df_month2.Weighted_Price.plot()
df_month2.forecast.plot(color='r', ls='--', label='Predicted Weighted_Price')
plt.legend()
plt.title('Bitcoin exchanges, by months')
plt.ylabel('mean USD')
plt.show()


```

![png](Arima.png)


##  T-Brain - 趨勢科技-台灣ETF價格預測競賽

![jpg](ETFStation.jpg)


