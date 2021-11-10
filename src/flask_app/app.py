from flask import Flask, render_template, request
import numpy as np
import sys
sys.path.append(r'.\src')
from models.predict_model import predict_with_trained_model

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict/', methods=['GET','POST'])
def predict():
    if request.method == "POST":
        #get form data
        first_measure = request.form.get('first_measure')
        measure_time_diff = request.form.get('measure_time_diff')
        heat_count = request.form.get('heat_count')
        mean_power_ratio = request.form.get('mean_power_ratio')

    try:
        pred = preprocessDataAndPredict(first_measure, 
              measure_time_diff, heat_count, mean_power_ratio)
        #pass prediction to template
        return render_template('predict.html', prediction = pred)
    except ValueError:
        return "Please Enter valid values"

def preprocessDataAndPredict(first_measure, 
              measure_time_diff, heat_count, mean_power_ratio):
    #keep all inputs in array
    test_data = [first_measure, 
              measure_time_diff, heat_count, mean_power_ratio]
    #convert value data into numpy array
    test_data = np.array(test_data)
    try:
        prediction = predict_with_trained_model(test_data)
        return prediction
    except ValueError:
        return 'Pred error'

if __name__ == '__main__':
    app.run(debug=True)    