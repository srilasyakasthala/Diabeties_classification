from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle

app=Flask(__name__)
model=pickle.load(open('model.pkl','rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method=='POST':
        val1 = request.form['pregnancies']
        val2 = request.form['glucose']
        val3 = request.form['bloodpressure']
        val4 = request.form['skinthickness']
        val5 = request.form['insulin']
        val6 = request.form['bmi']
        val7 = request.form['diabetespedigreefunction']
        val8 = request.form['age']
        arr = np.array([val5,val6])
        arr = arr.astype(np.float16) 
        pred=model.predict([arr])
        return render_template('index.html',data=pred) 
        



if __name__ == '__main__':
    app.run(debug=True)