import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from flask import Flask
from flask import request
from keras.models import load_model
import numpy as np


app = Flask(__name__)


N_time_stamps=100
N_features=9

model = load_model('har_model_left')



@app.route('/post', methods=["POST"])
def ppost():
    val=request.get_json()
    data = np.asarray(val, dtype= np.float32).reshape(1, N_time_stamps, N_features)
    result =  model.predict(data)
    final_result= result[0].round(decimals=2)
    maxx= final_result.argmax()
    data= '{"array": ['+str(final_result[0])+','+str(final_result[1])+','+str(final_result[2])+','+str(final_result[3])+','+str(final_result[4])+','+str(final_result[5])+','+str(final_result[6])+'],"max":'+str(maxx)+'}'
    print(data)
    return data


if __name__ == '__main__':
    app.run(host='0.0.0.0')# -*- coding: utf-8 -*-

#%%