from keras.applications import ResNet50
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from PIL import Image
import numpy as np
import flask
import io
from utils import pose_classification_utils as classifier
from utils import detector_utils
import cv2
import time
import base64
import re
from io import BytesIO
from flask_cors import CORS, cross_origin
import json
import tensorflow as tf

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
CORS(app, support_credentials=True)
pose_model = None
pose_classification_graph = None
pose_session = None
score_thresh = 0.6
pose_confidence_level = 0.6


def load_pose_model():
    # load keras model
    global pose_model, pose_classification_graph, pose_session
    pose_model, pose_classification_graph, pose_session = classifier.load_KerasGraph("hand_poses_wGarbage_10.h5")


def load_hand_model():
    print(">> loading frozen model for worker")
    detection_graph, sess = detector_utils.load_inference_graph()
    sess = tf.Session(graph=detection_graph)


def prepare_image(image, target):
    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")

    # resize the input image and preprocess it
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    # image = imagenet_utils.preprocess_input(image)

    # return the processed image
    return image


@app.route("/predict", methods=["POST"])
@cross_origin(supports_credentials=True)
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    starttime = time.time()
    data = {"success": False}
    global pose_model, pose_classification_graph, pose_session
    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.data:
            print(flask.request.data)
            # read the image in PIL format
            # image = flask.request.files["image"].read()
            # image = Image.open(io.BytesIO(image))
            json_data = json.loads(flask.request.data.decode('ascii'))
            image_data = re.sub('^data:image/.+;base64,', '', json_data['image'])
            image = Image.open(BytesIO(base64.b64decode(image_data)))
            image = img_to_array(image)
            # classify the input image
            data["result"] = classifier.classify(pose_model, pose_classification_graph, pose_session, image)
            if float(data["result"][max(data["result"], key = data["result"].get)]) > pose_confidence_level:
                data["highest"] = max(data["result"], key = data["result"].get)
            else:
                data["highest"] = None
            # indicate that the request was a success
            data["success"] = True
    endtime = time.time()
    print("Time Elapsed: " + str(endtime - starttime))
    # return the data dictionary as a JSON response
    return flask.jsonify(data)


# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
           "please wait until server has fully started"))
    load_pose_model()
    app.run()
