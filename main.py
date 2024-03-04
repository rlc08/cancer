from flask import Flask, render_template, request
import pandas as pd
import numpy as np  
import pickle

app = Flask(__name__)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)


def preprocess_input(data):

    input_data = pd.DataFrame(data, index=[0])

    input_data.replace('', np.nan, inplace=True)

    input_data.dropna(inplace=True)
    return input_data

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        form_data = request.form.to_dict()
        input_data = preprocess_input(form_data)
        if input_data.empty:
            return render_template('index.html', error='Please fill out all fields.')

        prediction = model.predict(input_data)
        return render_template('result.html', prediction=prediction[0])
    except Exception as e:
        return render_template('index.html', error='An error occurred. Please try again.')

if __name__ == '__main__':
    app.run(debug=True)
