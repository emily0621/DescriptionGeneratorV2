from flask import Flask, request, jsonify
from flask_cors import CORS
from model import AppModel
import base64
from io import BytesIO
from PIL import Image
import numpy as np

app = Flask(__name__)
CORS(app)

model = AppModel()


@app.route('/predict', methods=['POST'])
def predict():
    image = request.json.get('image')
    if 'data:' in image:
        image = image.split(',')[1]
    image_data = base64.b64decode(image)
    image_buffer = BytesIO(image_data)
    image = Image.open(image_buffer)
    return jsonify({'prediction': model.predict(np.array(image))[:-1]})


@app.route('/evaluate', methods=['POST'])
def evaluate():
    predicted = request.json.get('predicted')
    expected = request.json.get('expected')
    return model.evaluate(predicted, expected)


if __name__ == '__main__':
    app.run(debug=True)
