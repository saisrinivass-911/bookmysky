from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model
try:
    with open("flight_rf.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    expected_features = model.n_features_in_
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    expected_features = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template('index.html', prediction_text='Error: Model not loaded')

    try:
        # Extract inputs
        date_dep = request.form["Dep_Time"]
        date_arr = request.form["Arrival_Time"]
        Total_stops = int(request.form["Total_Stops"])
        airline = request.form["Airline"]
        source = request.form["Source"]
        destination = request.form["Destination"]

        # Process date-time inputs
        Journey_day = int(pd.to_datetime(date_dep).day)
        Journey_month = int(pd.to_datetime(date_dep).month)
        Dep_hour = int(pd.to_datetime(date_dep).hour)
        Dep_min = int(pd.to_datetime(date_dep).minute)
        Arrival_hour = int(pd.to_datetime(date_arr).hour)
        Arrival_min = int(pd.to_datetime(date_arr).minute)
        dur_hour = abs(Arrival_hour - Dep_hour)
        dur_min = abs(Arrival_min - Dep_min)

        # Airline Encoding (Ensure Only 11 Features)
        airlines = [
            'Jet Airways', 'IndiGo', 'Air India', 'Multiple carriers',
            'SpiceJet', 'Vistara', 'GoAir', 'Multiple carriers Premium economy',
            'Jet Airways Business', 'Vistara Premium economy', 'Trujet'
        ]
        airline_features = [1 if airline == name else 0 for name in airlines]
        if len(airline_features) != 11:
            return render_template('index.html', prediction_text="Error: Airline encoding incorrect")

        # Source Encoding (Ensure Only 4 Features)
        sources = ['Delhi', 'Kolkata', 'Mumbai', 'Chennai']
        source_features = [1 if source == name else 0 for name in sources]
        if len(source_features) != 4:
            return render_template('index.html', prediction_text="Error: Source encoding incorrect")

        # Destination Encoding (Ensure Only 4 Features)
        destinations = ['Cochin', 'Delhi', 'New Delhi', 'Hyderabad', 'Kolkata']
        destination_features = [1 if destination == name else 0 for name in destinations]
        
        # Fix Destination Encoding Issue (Ensure 4 Features)
        if len(destination_features) > 4:
            destination_features = destination_features[:4]  # Trim if extra
        elif len(destination_features) < 4:
            return render_template('index.html', prediction_text="Error: Destination encoding incorrect")

        # Create final feature array
        features = [
            Total_stops, Journey_day, Journey_month, Dep_hour, Dep_min,
            Arrival_hour, Arrival_min, dur_hour, dur_min
        ] + airline_features + source_features + destination_features

        # Debugging: Print feature length
        print(f"Feature count: {len(features)}, Expected: {expected_features}")

        # Convert to numpy array
        features = np.array(features).reshape(1, -1)

        # Ensure correct number of features
        if features.shape[1] != expected_features:
            return render_template('index.html', prediction_text=f'Error: Expected {expected_features} features, got {features.shape[1]}')

        # Make prediction
        prediction = model.predict(features)
        
        return render_template('index.html', prediction_text=f'Predicted Price: Rs. {prediction[0]:.2f}')
    
    except ValueError:
        return render_template('index.html', prediction_text='Error: Invalid input. Please enter correct values.')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)


