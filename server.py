from flask import Flask, request, redirect, Response
from Sliding_window import predict_image
from django.http import HttpResponse
app = Flask(__name__)
result = ''


@app.route('/upload', methods=['POST', 'GET'])
def upload():
    request.files['myfile'].save('uploaded.jpg')
    result = predict_image('uploaded.jpg')
    return Response(str(result))


app.run(debug=True)
