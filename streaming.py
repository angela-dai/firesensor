import Adafruit_DHT
import io
import picamera
import logging
import socketserver
from threading import Condition
from http import server
from time import sleep

DHT_SENSOR = Adafruit_DHT.DHT22
DHT_PIN = 22 #GPIO 22

PAGE="""\
<html>
<head>
<title>SENSE</title>
</head>
<body>
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
        # while True:
        #     humidity, temperature = Adafruit_DHT.read_retry(DHT_SENSOR, DHT_PIN)    
        #     if humidity is not None and temperature is not None:
        #         # humidity = str(humidity) + "%"
        #         # f = open('humidity.txt','w')
        #         # f.write(humidity)
        #         # f.close()
        #         temperature = str(temperature) + "C"
        #         f = open('temperature.txt','w')
        #         f.write(temperature)
        #         f.close()
        #     else:
        #         # f = open('humidity.txt','w')
        #         # f.write("ERROR")
        #         # f.close()
        #         f = open('temperature.txt','w')
        #         f.write("ERROR")
        #         f.close()
        #     sleep(1)
    finally:
        camera.stop_recording()

