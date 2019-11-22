from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import Adafruit_DHT
import argparse
import io
import picamera
import logging
import numpy as np
import socketserver
import time

from threading import Condition
from http import server
from PIL import Image, ImageFile
from tflite_runtime.interpreter import Interpreter

ImageFile.LOAD_TRUNCATED_IMAGES = True

DHT_SENSOR = Adafruit_DHT.DHT22
DHT_PIN = 22 #GPIO 22

PAGE="""\
<html>
<head>
<title>SENSE</title>
<style>
body { background: white; font-family: Arial, Helvetica, sans-serif;}
.row {display: flex;}
.column {flex: 50%;}
</style>
<script type="text/javascript">
function updateValue(){
var rawFile = new XMLHttpRequest();
rawFile = new XMLHttpRequest();
rawFile.open("GET", "/temperature.html", false);
rawFile.send(null);
document.getElementById("temperature").innerHTML = rawFile.responseText;
rawFile = new XMLHttpRequest();
rawFile.open("GET", "/humidity.html", false);
rawFile.send(null);
document.getElementById("humidity").innerHTML = rawFile.responseText;
rawFile = new XMLHttpRequest();
rawFile.open("GET", "/fireRisk.html", false);
rawFile.send(null);
document.getElementById("fireRisk").innerHTML = rawFile.responseText;
var rawFile = new XMLHttpRequest();
rawFile.open("GET", "/fireDetected.html", false);
rawFile.send(null);
document.getElementById("fireDetected").innerHTML = rawFile.responseText;
setTimeout('updateValue()',3000);
}
</script>
</head>
<body onLoad="updateValue()">
<div class="row">
<div class="column">
<img src="img/sense.png" width="141" height="48" />
</div>
<div class="column">
<h1>ONE MILE LAKE</h1>
</div>
</div>
<div class="row">
<div class="column">
<h2>Live View</h2>
<img src="stream.mjpg" width="640" height="480" />
</div>
<div class="column">
<h2>Current Details</h2>
<h3>Fire Risk:</h3>
<p id="fireRisk">LOW/MODERATE/HIGH/EXTREME</p>
<h3>Fire Detected:</h3>
<p id="fireDetected">YES/NO</p>
<h3>Temperature:</h3>
<p id="temperature">##</p>
<h3>Humidity:</h3>
<p id="humidity">##</p>
</div>
</div>
</body>
</html>
"""

def load_labels(path):
  with open(path, 'r') as f:
    return {i: line.strip() for i, line in enumerate(f.readlines())}


def set_input_tensor(interpreter, image):
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = image


def classify_image(interpreter, image, top_k=1):
  """Returns a sorted array of classification results."""
  set_input_tensor(interpreter, image)
  interpreter.invoke()
  output_details = interpreter.get_output_details()[0]
  output = np.squeeze(interpreter.get_tensor(output_details['index']))

  # If the model is quantized (uint8 data), then dequantize the results
  if output_details['dtype'] == np.uint8:
    scale, zero_point = output_details['quantization']
    output = scale * (output - zero_point)

  ordered = np.argpartition(-output, top_k)
  return [(i, output[i]) for i in ordered[:top_k]]


class StreamingOutput(object):
    def __init__(self):
        self.frame = None
        self.buffer = io.BytesIO()
        self.condition = Condition()

    def write(self, buf):
        if buf.startswith(b'\xff\xd8'):
            # New frame, copy the existing buffer's content and notify all
            # clients it's available
            self.buffer.truncate()
            with self.condition:
                self.frame = self.buffer.getvalue()
                self.condition.notify_all()
            self.buffer.seek(0)
        return self.buffer.write(buf)

class StreamingHandler(server.BaseHTTPRequestHandler):
    def do_GET(self):
        humidity, temperature = Adafruit_DHT.read_retry(DHT_SENSOR, DHT_PIN)
        if self.path == '/':
            self.send_response(301)
            self.send_header('Location', '/index.html')
            self.end_headers()
        elif self.path == '/index.html':
            content = PAGE.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Content-Length', len(content))
            self.end_headers()
            self.wfile.write(content)
        elif self.path == '/stream.mjpg':
            self.send_response(200)
            self.send_header('Age', 0)
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()
            try:
                while True:
                    with output.condition:
                        output.condition.wait()
                        frame = output.frame
                    self.wfile.write(b'--FRAME\r\n')
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', len(frame))
                    self.end_headers()
                    self.wfile.write(frame)
                    self.wfile.write(b'\r\n')
            except Exception as e:
                logging.warning(
                    'Removed streaming client %s: %s',
                    self.client_address, str(e))
        elif self.path == '/temperature.html':
            if temperature is not None:
                temperature = str(temperature)+"°C"
                content = temperature.encode('utf-8')
                self.send_response(200)
                self.send_header('Content-Type', 'text/html')
                self.send_header('Content-Length', len(content))
                self.end_headers()
                self.wfile.write(content)
        elif self.path == '/humidity.html':
            if humidity is not None:
                humidity = str(humidity)+"%"
                content = humidity.encode('utf-8')
                self.send_response(200)
                self.send_header('Content-Type', 'text/html')
                self.send_header('Content-Length', len(content))
                self.end_headers()
                self.wfile.write(content)
        elif self.path == '/fireRisk.html':
            if humidity is not None and temperature is not None:
                risk = ""
                tempRisk = ""
                humidityRisk = ""
                if temperature > 35:
                    tempRisk = "EXTREME"
                elif temperature > 30:
                    tempRisk = "HIGH"
                elif temperature > 25:
                    tempRisk = "MODERATE"
                else:
                    tempRisk = "LOW"
                if humidity < 15: 
                    humidityRisk = "EXTREME"
                elif humidity < 25:
                    humidityRisk = "HIGH"
                elif humidity < 35:
                    humidityRisk = "MODERATE"
                else:
                    humidityRisk = "LOW"
                if tempRisk == "EXTREME" and humidity == "EXTREME":
                    risk = "EXTREME"
                elif tempRisk == "EXTREME" or humidityRisk == "EXTREME":
                    risk = "HIGH"
                elif temperature == "MODERATE" or humidityRisk == "MODERATE":
                    risk = "MODERATE"
                else:
                    risk = "LOW"
                content = risk.encode('utf-8')
                self.send_response(200)
                self.send_header('Content-Type', 'text/html')
                self.send_header('Content-Length', len(content))
                self.end_headers()
                self.wfile.write(content)
        elif self.path == '/fireDetected.html':
            labels = load_labels("model/labels.txt")
            interpreter = Interpreter("model/model_unquant.tflite")
            interpreter.allocate_tensors()
            _, height, width, _ = interpreter.get_input_details()[0]['shape']
            # image = Image.open(output.buffer).save("picture", "JPEG")
            image = Image.open(output.buffer).convert('RGB').resize((width, height),
                                                         Image.ANTIALIAS)
            results = classify_image(interpreter, image)
            label_id, prob = results[0]
            result = str(labels[label_id])+str(prob)
            content = result.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Content-Length', len(content))
            self.end_headers()
            self.wfile.write(content)
        else:
            self.send_error(404)
            self.end_headers()

class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True

with picamera.PiCamera(resolution='640x480', framerate=24) as camera:
    output = StreamingOutput()
    camera.start_recording(output, format='mjpeg')
    try:
        address = ('', 8000)
        server = StreamingServer(address, StreamingHandler)
        server.serve_forever()
    finally:
        camera.stop_recording()

