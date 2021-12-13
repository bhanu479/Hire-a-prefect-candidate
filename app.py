from flask import Flask, render_template, request
import jsonify
import requests
from joblib import dump, load
import pandas as pd
import convertor
from tika import parser
def data_creater(data):
    #Converting the data to pdf
    raw = parser.from_file(data)
    x = " ".join(raw['content'].strip().split('\n')[1:])
    #Cleaning data
    x = convertor.cleanResume(x)
    x=convertor.data_corrector(x)
    return x


app = Flask(__name__)
model = load('src.joblib')
@app.route('/',methods=['GET'])
def Home():
    return render_template('job.html')


@app.route("/predict", methods=['POST','GET'])
def predict():
    if request.method == 'POST':
        data = request.form['data']
        data=data_creater(data)
        result = model.predict(data)

        if result < 0:
            return render_template('job.html', prediction_text="Sorry you cannot sell this house")
        else:
            return render_template('job.html', prediction_text="You Can Sell The House at {} in 1000$".format(result))

    else:
        return render_template('job.html')

if __name__=="__main__":
    app.run(debug=True)
