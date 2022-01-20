from flask import Flask, request, redirect, Response
from Sliding_window import predict_image
from django.http import HttpResponse
app = Flask(__name__)
result = ''


@app.route('/classify', methods=['POST', 'GET'])
def classify():
    request.files['myfile'].save('uploaded.jpg')
    result = predict_image('uploaded.jpg')
    print(result)
    return Response(str(result))


app.run(debug=True)
