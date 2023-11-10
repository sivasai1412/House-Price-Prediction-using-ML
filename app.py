
from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

import model # load model.py


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    
    #take data from form and store in each feature    
    input_features = [x for x in request.form.values()]
    bath = input_features[0]
    balcony = input_features[1]
    total_sqft_int = input_features[2]
    bhk = input_features[3]
    price_per_sqft = input_features[4]
     
    # predict the price of house by calling model.py
    predicted_price = model.predict_house_price(bath,balcony,total_sqft_int,bhk,price_per_sqft)       
 
 
    # render the html page and show the output
    return render_template('index.html', prediction_text='Predicted Price of House is {:.2f}'.format(predicted_price))

if __name__ == "__main__":
    app.run(debug=True)
