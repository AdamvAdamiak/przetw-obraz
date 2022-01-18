import json
from os import path
from flask import Flask, request, Response, abort, redirect, render_template
import requests as r
from werkzeug.wrappers import response
import numpy as np
app = Flask(__name__)

REDIRECT_URL = "http://127.0.0.1:5500/frontend/a.html"

class Server: 
    
    @app.route('/upload', methods=['GET'])
    def upload():
        img = request.args.get('image')
        print(np.array(img))
        print(type(img))
        return redirect(REDIRECT_URL, code=302)

if __name__ == "__main__":
    app.run()