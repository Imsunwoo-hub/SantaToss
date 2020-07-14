import base64
import json
import os
from io import BytesIO
import re
from flask import Flask, request, jsonify
import flask
# from keras.preprocessing import image
from predict import predict
from flask_cors import CORS

# from datasets import preprocessing
# import tensorflow as tf
import time

app = Flask(__name__)


# Uncomment this line if you are making a Cross domain request
CORS(app)


# Testing URL
@app.route('/image/', methods=['GET', 'POST'])
def hello_world():
    return jsonify({'text':'Hello, World!'})


@app.route('/image/predict/', methods=['POST'])
def image_classifier():
    # Decoding and pre-processing base64 image
    final_res='';
    filename='';
    create_name=time.strftime('%c', time.localtime(time.time()));
    # create_name='someImage';
    b64 = flask.request.form['b64']
    img_name = flask.request.form['img_name']

    if not b64=="null": 
        b64 = re.sub('^data:image/.+;base64,', '', b64)
        byte_data = base64.b64decode(b64)
        img = BytesIO(byte_data)

        imgdata = byte_data
        filename = '/home/ubuntu/final/images/'+create_name+'.jpg';
        with open(filename, 'wb') as f:
            f.write(imgdata)

    else: 
        filename = '/home/ubuntu/final/random_images/' + img_name



    final_res=predict(filename)

    if not b64=="null":
        os.remove(filename)
    return jsonify({'caption': '\n'.join(final_res)})

#

if __name__ == "__main__":
    app.run(host='0.0.0.0')
#
