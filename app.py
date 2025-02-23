from flask import Flask, render_template, request, jsonify, url_for
import pickle
import pandas as pd
import numpy as np
import plotly
import plotly.graph_objs as go
import json
from datetime import datetime

def format_number(value):
    return "{:,.2f}".format(value)

app = Flask(__name__)
app.jinja_env.filters['format_number'] = format_number

# Load model and data
model_data = pickle.load(open('data/crop_prediction_model.pkl', 'rb'))
model = model_data['model']
label_encoder_soil = model_data['label_encoder_soil']
label_encoder_period = model_data['label_encoder_period']
label_encoder_crop = model_data['label_encoder_crop']
scaler = model_data['scaler']

# Define feature names
feature_names = ['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 
                 'Humidity', 'pH', 'Soil Type', 'Period of Month', 
                 'NPK_Ratio', 'Temp_Humidity_Index', 'Soil_Moisture_Index', 
                 'NPK_TH_Index']

# Crop data dictionary with yield and price information
CROP_DATA = {
     'Almond': {'yield_per_acre': 2, 'price_per_kg': 15000.0},
    'Arecanut': {'yield_per_acre': 2, 'price_per_kg': 6000.0},
    'Banana': {'yield_per_acre': 200, 'price_per_kg': 1500.0},
    'Black Pepper': {'yield_per_acre': 35, 'price_per_kg': 20000.0},
    'Bottle Gourd': {'yield_per_acre': 120, 'price_per_kg': 1000.0},
    'Brinjal': {'yield_per_acre': 100, 'price_per_kg': 1200.0},
    'Cabbage': {'yield_per_acre': 200, 'price_per_kg': 800.0},
    'Cardamom': {'yield_per_acre': 3, 'price_per_kg': 80000.0},
    'Carrot': {'yield_per_acre': 120, 'price_per_kg': 1500.0},
    'Cashew Nut': {'yield_per_acre': 2, 'price_per_kg': 12000.0},
    'Castor Seed': {'yield_per_acre': 8, 'price_per_kg': 5000.0},
    'Cauliflower': {'yield_per_acre': 150, 'price_per_kg': 1200.0},
    'Chickpea': {'yield_per_acre': 12, 'price_per_kg': 4500.0},
    'Chilli': {'yield_per_acre': 15, 'price_per_kg': 15000.0},
    'Coconut': {'yield_per_acre': 80, 'price_per_kg': 25.0},
    'Coffee': {'yield_per_acre': 3, 'price_per_kg': 25000.0},
    'Colocasia': {'yield_per_acre': 80, 'price_per_kg': 1000.0},
    'Coriander': {'yield_per_acre': 15, 'price_per_kg': 6000.0},
    'Cotton': {'yield_per_acre': 8, 'price_per_kg': 7000.0},
    'Cucumber': {'yield_per_acre': 100, 'price_per_kg': 800.0},
    'Garlic': {'yield_per_acre': 80, 'price_per_kg': 3000.0},
    'Ginger': {'yield_per_acre': 100, 'price_per_kg': 4000.0},
    'Gram': {'yield_per_acre': 10, 'price_per_kg': 4000.0},
    'Grapes': {'yield_per_acre': 80, 'price_per_kg': 2500.0},
    'Green Peas': {'yield_per_acre': 100, 'price_per_kg': 1500.0},
    'Groundnut': {'yield_per_acre': 15, 'price_per_kg': 5000.0},
    'Guava': {'yield_per_acre': 120, 'price_per_kg': 1200.0},
    'Jowar': {'yield_per_acre': 8, 'price_per_kg': 2500.0},
    'Lemon': {'yield_per_acre': 120, 'price_per_kg': 1500.0},
    'Linseed': {'yield_per_acre': 10, 'price_per_kg': 5000.0},
    'Maize': {'yield_per_acre': 80, 'price_per_kg': 1500.0},
    'Mango': {'yield_per_acre': 100, 'price_per_kg': 2000.0},
    'Masoor Dal': {'yield_per_acre': 12, 'price_per_kg': 4000.0},
    'Methi': {'yield_per_acre': 12, 'price_per_kg': 6000.0},
    'Mustard': {'yield_per_acre': 10, 'price_per_kg': 4500.0},
    'Nutmeg': {'yield_per_acre': 4, 'price_per_kg': 30000.0},
    'Onion': {'yield_per_acre': 100, 'price_per_kg': 1500.0},
    'Orange': {'yield_per_acre': 100, 'price_per_kg': 2000.0},
    'Paddy': {'yield_per_acre': 30, 'price_per_kg': 2500.0},
    'Papaya': {'yield_per_acre': 120, 'price_per_kg': 1500.0},
    'Peas': {'yield_per_acre': 120, 'price_per_kg': 1500.0},
    'Pineapple': {'yield_per_acre': 150, 'price_per_kg': 2000.0},
    'Pomegranate': {'yield_per_acre': 120, 'price_per_kg': 3000.0},
    'Potato': {'yield_per_acre': 250, 'price_per_kg': 1000.0},
    'Pumpkin': {'yield_per_acre': 100, 'price_per_kg': 900.0},
    'Ragi': {'yield_per_acre': 8, 'price_per_kg': 3000.0},
    'Rajma': {'yield_per_acre': 10, 'price_per_kg': 4500.0},
    'Red Gram': {'yield_per_acre': 8, 'price_per_kg': 6000.0},
    'Ridge Gourd': {'yield_per_acre': 100, 'price_per_kg': 900.0},
    'Rubber': {'yield_per_acre': 6, 'price_per_kg': 20000.0},
    'Sesame': {'yield_per_acre': 8, 'price_per_kg': 6500.0},
    'Soybean': {'yield_per_acre': 10, 'price_per_kg': 4000.0},
    'Spinach': {'yield_per_acre': 150, 'price_per_kg': 1200.0},
    'Sponge Gourd': {'yield_per_acre': 100, 'price_per_kg': 800.0},
    'Sweet Corn': {'yield_per_acre': 120, 'price_per_kg': 2000.0},
    'Tomato': {'yield_per_acre': 150, 'price_per_kg': 1200.0},
    'Watermelon': {'yield_per_acre': 80, 'price_per_kg': 600.0}

}

def create_pie_chart(crops, probabilities):
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    
    fig = go.Figure(data=[go.Pie(
        labels=crops,
        values=probabilities,
        hole=.3,
        marker=dict(colors=colors),
        textinfo='label+percent',
        textfont=dict(size=14)
    )])
    
    fig.update_layout(
        title="Crop Distribution Analysis",
        title_x=0.5,
        title_font=dict(size=20, color='white'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        margin=dict(t=50, l=0, r=0, b=0),
        height=400
    )
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_revenue_chart(revenue_data):
    crops = [data['crop'].title() for data in revenue_data]
    revenues = [data['gross_revenue'] for data in revenue_data]
    
    fig = go.Figure(data=[
        go.Bar(
            x=crops,
            y=revenues,
            marker_color='#2ecc71'
        )
    ])
    
    fig.update_layout(
        title="Expected Revenue by Crop",
        title_x=0.5,
        xaxis_title="Crops",
        yaxis_title="Expected Revenue (₹)",
        title_font=dict(size=20, color='white'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        margin=dict(t=50, l=50, r=50, b=50),
        height=400
    )
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def calculate_revenue(crop_name):
    crop_name = crop_name.title()
    if crop_name in CROP_DATA:
        crop_info = CROP_DATA[crop_name]
        yield_per_acre = crop_info['yield_per_acre']
        price_per_kg = crop_info['price_per_kg']
        gross_revenue = yield_per_acre * price_per_kg
        return {
            'crop': crop_name,
            'yield_per_acre': yield_per_acre,
            'price_per_kg': price_per_kg,
            'gross_revenue': gross_revenue
        }
    return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if request.is_json:
            data = request.json
        else:
            data = request.form.to_dict()

        # Extract base features
        input_data = {
            'Nitrogen': float(data['Nitrogen']),
            'Phosphorus': float(data['Phosphorus']),
            'Potassium': float(data['Potassium']),
            'Temperature': float(data['Temperature']),
            'Humidity': float(data['Humidity']),
            'pH': float(data['pH']),
            'Soil Type': data['Soil Type'],
            'Period of Month': data['Period of Month']
        }

        # Calculate derived features
        input_data['NPK_Ratio'] = (input_data['Nitrogen'] + input_data['Phosphorus'] + input_data['Potassium']) / 3
        input_data['Temp_Humidity_Index'] = input_data['Temperature'] * input_data['Humidity'] / 100
        input_data['Soil_Moisture_Index'] = input_data['Humidity'] * input_data['pH'] / 100
        input_data['NPK_TH_Index'] = input_data['NPK_Ratio'] * input_data['Temp_Humidity_Index'] / 100

        # Create DataFrame and encode categorical variables
        df = pd.DataFrame([input_data])
        df['Soil Type'] = label_encoder_soil.transform(df['Soil Type'])
        df['Period of Month'] = label_encoder_period.transform(df['Period of Month'])
        
        # Ensure correct feature order
        df = df[feature_names]
        
        # Scale features
        scaled_features = scaler.transform(df)
        
        # Make prediction
        prediction = model.predict(scaled_features)
        probabilities = model.predict_proba(scaled_features)[0]
        
        # Get top 3 predictions
        top_3_idx = probabilities.argsort()[-3:][::-1]
        top_3_crops = label_encoder_crop.inverse_transform(top_3_idx)
        top_3_probs = probabilities[top_3_idx] * 100
        
        # Calculate revenue for top crops
        revenue_data = []
        for crop in top_3_crops:
            rev_data = calculate_revenue(crop)
            if rev_data:
                revenue_data.append(rev_data)
        
        # Create visualizations
        pie_chart = create_pie_chart(top_3_crops, top_3_probs)
        revenue_chart = create_revenue_chart(revenue_data)
        
        result = {
            'predicted_crop': label_encoder_crop.inverse_transform(prediction)[0],
            'confidence': float(max(probabilities) * 100),
            'top_3_crops': [
                {'crop': crop, 'probability': prob} 
                for crop, prob in zip(top_3_crops, top_3_probs)
            ],
            'revenue_data': revenue_data,
            'pie_chart': pie_chart,
            'revenue_chart': revenue_chart,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        app.config['LAST_PREDICTION'] = result
        
        if request.is_json:
            return jsonify(result)
        return render_template('result.html', result=result)
        
    except Exception as e:
        error_msg = f"Prediction Error: {str(e)}"
        if request.is_json:
            return jsonify({'error': error_msg}), 400
        return render_template('index.html', error=error_msg)

@app.route('/yield-calculator')
def yield_calculator():
    last_prediction = app.config.get('LAST_PREDICTION', {})
    return render_template('yield_calculator.html', prediction=last_prediction)

@app.route('/calculate-profit', methods=['POST'])
def calculate_profit():
    try:
        costs = {
            'seed_cost': float(request.form['seed_cost']),
            'land_cost': float(request.form['land_cost']),
            'fertilizer_cost': float(request.form['fertilizer_cost']),
            'irrigation_cost': float(request.form['irrigation_cost']),
            'labor_cost': float(request.form['labor_cost'])
        }
        
        total_cost = sum(costs.values())
        last_prediction = app.config.get('LAST_PREDICTION', {})
        revenue_data = last_prediction.get('revenue_data', [])
        
        profits = []
        for crop_revenue in revenue_data:
            if crop_revenue:
                profit = {
                    'crop': crop_revenue['crop'],
                    'gross_revenue': crop_revenue['gross_revenue'],
                    'total_cost': total_cost,
                    'net_profit': crop_revenue['gross_revenue'] - total_cost
                }
                profits.append(profit)
        
        return render_template('profit_result.html', 
                             profits=profits, 
                             costs=costs,
                             calculation_date=datetime.now().strftime("%Y-%m-%d"))
                             
    except Exception as e:
        return render_template('yield_calculator.html', 
                             error=f"Calculation Error: {str(e)}",
                             prediction=app.config.get('LAST_PREDICTION', {}))

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
