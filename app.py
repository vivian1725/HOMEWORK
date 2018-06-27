from flask import Flask, request, abort
import json
import requests
import pandas as pd
import numpy as np
import time


from linebot import (
    LineBotApi, WebhookHandler
)
from linebot.exceptions import (
    InvalidSignatureError
)
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage, ImageSendMessage
)

app = Flask(__name__)

# Channel Access Token
line_bot_api = LineBotApi('WR0Yo1y3SLRGXbsYd3SifovyVzFWZhnBlZVIC5jWCbrKfyx1obVgcQ+zFw9Mg0kWj0e6VeDDCjTNprUyG584rIhWTn8/ELUgkoBXy5h8ejQT9SuZwvs7z76rkrFhfp7L0OMnKUyKmLcRQW2RMX5jBAdB04t89/1O/w1cDnyilFU=')
# Channel Secret
handler = WebhookHandler('d495c7f8877200249676da9dfdd3baf8')

# 監聽所有來自 /callback 的 Post Request
@app.route("/callback", methods=['POST'])
def callback():
    # get X-Line-Signature header value
    signature = request.headers['X-Line-Signature']

    # get request body as text
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)

    # handle webhook body
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)

    return 'OK'



@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    if event.message.text == "Bitcon price" :
        message = TextSendMessage(text=bitconprice())
        line_bot_api.reply_message(
             event.reply_token,
             message)
    elif event.message.text == "Bitcon chart" :
        message = ImageSendMessage(
        original_content_url='https://i.imgur.com/DKJzW0l.png',
        preview_image_url='https://i.imgur.com/DKJzW0l.png')
        line_bot_api.reply_message(event.reply_token, message)

def bitconprice():
    endpoint = 'https://min-api.cryptocompare.com/data/histoday'
    res = requests.get(endpoint + '?fsym=BTC&tsym=USD&limit=7')
    hist = pd.DataFrame(json.loads(res.content)['Data'])
    hist = hist.set_index('time')
    hist.index = pd.to_datetime(hist.index, unit='s')
    nowdate=time.strftime("%Y-%m-%d", time.localtime())
    CUR=hist[hist.index >= nowdate]
    price=CUR['close'].values
    price=price[0]
    return price

#def bitconchart():
 #   endpoint = 'https://min-api.cryptocompare.com/data/histoday'
 #   res = requests.get(endpoint + '?fsym=BTC&tsym=USD&limit=300')
 #   hist = pd.DataFrame(json.loads(res.content)['Data'])
 #   hist = hist.set_index('time')
 #   hist.index = pd.to_datetime(hist.index, unit='s')
 #   hist['ma7']=hist['close'].rolling(window=7).mean()
 #   CUR=hist[hist.index >= '2018-01-01']
 #   plt.plot(CUR[['close','ma7']])
 #   git=plt.gcf()
 #   plt.show()
 #   git.savefig('chart.png')
    

import os
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
