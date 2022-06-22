# khởi tạo web
from cv2 import resize
from flask import Flask
# Xử lý yêu cầu
from flask import request, jsonify
# Gọi API từ client
from flask_cors import CORS, cross_origin

import os
os.environ['CUDA_VISIBLE_DEVICE'] ='-1'
import tensorflow as tf
import numpy as np
from keras.models import load_model
import cv2
import sys

# init
session = tf.compat.v1.Session()
graph = tf.compat.v1.get_default_graph()

# Định nghĩa class
class_name = ['NORMAL','PNEUMONIA']
# Load models
with session.as_default():
    with graph.as_default():
        my_model = load_model("model .h5")

# Tạo API HTTP server
app = Flask(__name__)
CORS(app)
app.config['COR_HEADERS'] = 'Content-Type'
 

# Truy cập, kiểm tra xem server đã hoạt động hay chưa!
@app.route('/')
@cross_origin(origins='*')
def index():
    return "Server is running"

@app.route('/upload',methods=['POST'])
@cross_origin(origins='*')
def upload():
    global session, graph, my_model
    #  Receive Input and Output
    # Receive files and convert to img
    f = request.files['file']
    image = cv2.imdecode(np.fromstring(f.read(), np.uint8), cv2.IMREAD_COLOR)

    # Predict img
    image = cv2.resize(image, dsize=(128,128))
    # Convert to tensorflow
    image = np.expand_dims(image, axis=0)

    with session.as_default():
        with graph.as_default():
            predict = my_model.predict(image)
    # Return to client
    print("This picture is", class_name[np.argmax(predict)])
    #f.save(secure_filename(f.filename))
    return class_name[np.argmax(predict)]


if __name__ == '__main__':
    app.run(debug = True, port = 8000)