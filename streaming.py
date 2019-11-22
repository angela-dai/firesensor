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
from PIL import Image
from tflite_runtime.interpreter import Interpreter

image = ""

DHT_SENSOR = Adafruit_DHT.DHT22
DHT_PIN = 22 #GPIO 22

PAGE="""\
<html>
<head>
<title>SENSE</title>
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
rawFile.open("GET", "/fireDetected.html", false);
rawFile.send(null);
document.getElementById("fireDetected").innerHTML = rawFile.responseText;
setTimeout('updateValue()',100);
}
</script>
</head>
<body onLoad="updateValue()">
<h1>ONE MILE LAKE</h1>
<h2>Live View</h2>
<img src="stream.mjpg" width="640" height="480" />
<h2>Current Details</h2>
<h3>Fire Risk:</h3>
<p id="fireRisk">LOW/MODERATE/HIGH/EXTREME</p>
<h3>Fire Detected:</h3>
<p id="fireDetected">YES/NO</p>
<h3>Temperature:</h3>
<p id="temperature">##</p>
<h3>Humidity:</h3>
<p id="humidity">##</p>
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

    def classify_image(interpreter, top_k=1):
        """Returns a sorted array of classification results."""
        image = Image.open(stream).convert('RGB').resize((width, height),
                                                         Image.ANTIALIAS)
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

class StreamingHandler(server.BaseHTTPRequestHandler):
    def do_GET(self):
        humidity, temperature = Adafruit_DHT.read_retry(DHT_SENSOR, DHT_PIN)
        output = StreamingOutput()
        if self.path == '/':
            self.send_response(301)
            self.send_header('Location', '/index.html')
            self.end_headers()
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
        elif self.path == '/fireDetected.html':
            if image != "":
                labels = load_labels("model/labels.txt")
                interpreter = Interpreter("model/model.tflite")
                interpreter.allocate_tensors()
                _, height, width, _ = interpreter.get_input_details()[0]['shape']
                results = output.classify_image(interpreter)
                label_id, prob = results[0]
                result = str(labels[label_id])+str(prob)
                content = result.encode('utf-8')
                self.send_response(200)
                self.send_header('Content-Type', 'text/html')
                self.send_header('Content-Length', len(content))
                self.end_headers()
                self.wfile.write(content)
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

