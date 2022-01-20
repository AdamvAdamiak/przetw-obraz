from flask import Flask, request, redirect
from Sliding_window import predict_image

app = Flask(__name__)


@app.route('/upload', methods=['POST', 'GET'])
def show_user():
    request.files['myfile'].save('uploaded.jpg')
    print(predict_image('uploaded.jpg'))
    return redirect('http://127.0.0.1:5500/frontend/index.html')


app.run(debug=True)

