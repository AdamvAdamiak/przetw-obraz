from flask import Flask, request, redirect, Response
from Sliding_window import predict_image
from django.http import HttpResponse
app = Flask(__name__)
result = ''


@app.route('/classify', methods=['POST', 'GET'])
def classify():
    request.files['myfile'].save('uploaded.jpg')
    result = predict_image('uploaded.jpg')
    return Response(str(result))

@app.route('/classifydigit', methods=['POST', 'GET'])
def classifydigit():
    request.files['myfile'].save('uploaded.jpg')
    result = predict_image('uploaded.jpg','D')
    return Response(str(result))

@app.route('/classifyletter', methods=['POST', 'GET'])
def classifyletter():
    request.files['myfile'].save('uploaded.jpg')
    result = predict_image('uploaded.jpg','L')
    return Response(str(result))

app.run(debug=True)
