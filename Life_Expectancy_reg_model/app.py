from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from difflib import get_close_matches
import pickle

app = Flask(__name__)
CORS(app)

# Globals for data and model
df = None
model = None
available_country_columns = None
available_countries = None

def normalize_country_name(name):
    return name.strip().lower().replace(' ', '_')

def load_artifacts():
    global df, model, available_country_columns, available_countries
    
    # Load cleaned/preprocessed DataFrame (You can save as pickle or CSV)
    df = pd.read_pickle('./artifacts/cleaned_life_expectancy_df.pkl')
    
    # Load trained model
    with open('./artifacts/life_expectancy_model.pkl', 'rb') as f:
        model = pickle.load(f)

    # Get all country columns for one-hot encoding
    available_country_columns = [col for col in df.columns if col.startswith('Country_')]
    available_countries = [col.replace('Country_', '').replace('_', ' ') for col in available_country_columns]

@app.route('/get_country_names', methods=['GET'])
def get_country_names():
    # Return country names capitalized nicely
    countries = [c.title() for c in available_countries]
    return jsonify({'countries': countries})

@app.route('/predict_life_expectancy', methods=['POST'])
def predict_life_expectancy():
    data = request.get_json()
    if not data or 'country' not in data or 'year' not in data:
        return jsonify({'error': 'Invalid input'}), 400
    
    country_input = data['country']
    year_input = data['year']
    
    normalized_input = normalize_country_name(country_input)
    normalized_available = [normalize_country_name(c) for c in available_countries]

    matched_country = None
    matched_column = None

    # Exact match
    for col, country in zip(available_country_columns, available_countries):
        if normalize_country_name(country) == normalized_input:
            matched_country = country
            matched_column = col
            break

    # Close match if no exact match
    if not matched_country:
        close_matches = get_close_matches(normalized_input.replace('_', ' '),
                                          [c.replace('_', ' ') for c in normalized_available],
                                          n=3, cutoff=0.6)
        if close_matches:
            # Pick best match automatically (could be improved by UI)
            matched_country = close_matches[0]
            matched_column = f"Country_{matched_country.replace(' ', '_')}"
            matched_country = matched_country.title()
        else:
            return jsonify({'error': f"Country '{country_input}' not recognized."}), 400
    
    # Filter data for matched country
    country_data = df[df[matched_column] == 1]

    if country_data.empty:
        return jsonify({'error': f"No historical data found for {matched_country}."}), 400
    
    # Take latest row and update year
    latest_row = country_data.sort_values('Year', ascending=False).head(1).copy()
    latest_row['Year'] = year_input

    # Predict log life expectancy, then exponentiate
    log_prediction = model.predict(latest_row)[0]
    prediction = np.exp(log_prediction)

    return jsonify({
        # 'country': matched_country,
        # 'year': year_input,
        'estimated_life_expectancy': round(prediction, 2)
    })

if __name__ == '__main__':
    print("Loading artifacts...")
    load_artifacts()
    print("Artifacts loaded. Starting Flask server...")
    app.run(debug=True)
