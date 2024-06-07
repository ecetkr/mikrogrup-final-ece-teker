from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
from notebooks.preprocessing import preprocess_data

app = Flask(__name__)

# Load the trained model
model = joblib.load("data/XGBoost_best_model.joblib")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get the input values from the form
    input_data = {
        'Number_of_Customers': [float(request.form["Number_of_Customers"])],
        'Menu_Price': [float(request.form["Menu_Price"])],
        'Marketing_Spend': [float(request.form["Marketing_Spend"])],
        'Cuisine_Type': [request.form["Cuisine_Type"]],
        'Average_Customer_Spending': [float(request.form["Average_Customer_Spending"])],
        'Promotions': [float(request.form["Promotions"])],
        'Reviews': [float(request.form["Reviews"])]
    }
    
    # Create a DataFrame with the input data
    input_df = pd.DataFrame(input_data)
    
    # Make prediction using the loaded model
    prediction = model.predict(input_df)
    
    # Format the prediction as a string
    predicted_revenue = f"${prediction[0]:,.2f}"
    
    # Render the result template with the prediction
    return render_template("result.html", prediction=predicted_revenue)

if __name__ == "__main__":
    app.run(debug=True)
