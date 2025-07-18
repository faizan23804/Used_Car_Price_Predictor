from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

application = Flask(__name__)  # Changed from 'app' to 'application'

# Load model and transformer
model = joblib.load('random_forest_regressor.pkl')
transformer = joblib.load('preprocessor.pkl')

@application.route('/', methods=['GET', 'POST'])  # Use 'application' instead of 'app'
def index():
    if request.method == 'POST':
        try:
            # Read form inputs
            car_name = request.form['car_name']
            seller_type = request.form['seller_type']
            fuel_type = request.form['fuel_type']
            transmission_type = request.form['transmission_type']
            vehicle_age = int(request.form['vehicle_age'])
            km_driven = int(request.form['km_driven'])
            mileage = float(request.form['mileage'])
            engine = int(request.form['engine'])
            max_power = float(request.form['max_power'])
            seats = int(request.form['seats'])

            # Create DataFrame for model input
            input_df = pd.DataFrame([[car_name, seller_type, fuel_type, transmission_type,
                                      vehicle_age, km_driven, mileage, engine, max_power, seats]],
                columns=['car_name', 'seller_type', 'fuel_type', 'transmission_type',
                         'vehicle_age', 'km_driven', 'mileage', 'engine', 'max_power', 'seats'])

            # Preprocess
            transformed_input = transformer.transform(input_df)

            # Predict
            prediction = model.predict(transformed_input)[0]
            prediction = round(prediction, 2)

            return render_template('id.html', result=f"Estimated Selling Price: ₹{prediction:,.2f}")
        except Exception as e:
            return render_template('id.html', result=f"Error: {str(e)}")

    return render_template('id.html')

if __name__ == '__main__':
    application.run(debug=True)
