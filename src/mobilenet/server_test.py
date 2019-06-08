from flask import Flask, flash, redirect, render_template, \
     request, url_for

import base64
from classifier import single_imgae_classifier
import os
import settings as st
import json
import socket


app = Flask(__name__)

"""
$ export FLASK_APP=hello.py
$ flask run
 * Running on http://127.0.0.1:5000/
"""

@app.route('/')
def index():
    return "this is where to start"

@app.route('/initialize', methods=['GET','POST'])
def initialize():
    print("initial request received")
    local_data = read_from_local_data('Landmarks.json')
    local_data = local_data['Landmarks']
    result = {}
    result['coordinate'] = local_data['melbourne central']['coordinate']
    result['local data'] = local_data
    json_result = json.dumps(result)
    return json_result

@app.route('/receive', methods=['GET','POST'])
def receive():
    print("a image received")
    image_data = request.get_json()
    filename = save_img_from_client(image_data)
    img = os.path.join(st.MOBILENET_ROOT, filename)
    predicted_class = single_imgae_classifier(img)[0]

    local_data = read_from_local_data('Landmarks.json')
    local_data = local_data['Landmarks']
    local_data[predicted_class]['category'] = 'label'

    result = {}
    result['label'] = predicted_class
    result['coordinate'] = local_data[predicted_class]['coordinate']
    result['local data'] = local_data

    json_result = json.dumps(result)
    return json_result


    # imageFile = flask.request.files.get('image','') // return None


def save_img_from_client(image_data):
    image_data = image_data['image']
    image_data = base64.b64decode(image_data)
    filename = "uploadImg.jpg"
    file = open(filename, 'wb')
    file.write(image_data)
    return filename

def read_from_local_data(jsonfile):
    file = open(jsonfile, 'r')
    datastore = json.load(file)
    return datastore

def get_ip_addr():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(('8.8.8.8', 80))
    ip = s.getsockname()[0]
    s.close()
    return ip

if __name__ == '__main__':
    app.run(host=get_ip_addr(),port=5000,debug=False)