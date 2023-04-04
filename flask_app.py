# Image transfer imports
from flask import Flask, request
import json
from PIL import Image
from io import BytesIO
import base64

# CORS import
from flask_cors import CORS

# Classifier import
from hockey_classifier import Hockey_classifier

# Flask app
app = Flask(__name__)
CORS(app)

# Classes data files
teams_filename = "hockey.txt"
sports_filename = "sports.txt"

# NHL Classifier
classifier = Hockey_classifier(teams_filename, sports_filename)

@app.route('/')
def hello():
    return 'Hello World! -Hockey team detection'

@app.route("/generate-hockey-team-label/", methods=["GET", "POST"])
def handle_request():
    """
    Handles the reception of the request from the client and returns a response
    """
    data = request.get_json()
    base64_str = data["image"]
    image_data = base64.b64decode(base64_str)
    img = Image.open(BytesIO(image_data))
    prediction = classifier.classify(img)
    return json.dumps({"prediction": prediction[0] if prediction[0] else ""}, separators=(',', ':')), 200

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')