from flask import Flask, render_template, request
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

app = Flask(__name__)

# Load the trained model
model = joblib.load("data/XGBoost_best_model.joblib")

# Define preprocessing steps
numeric_features = ['Number_of_Customers', 'Menu_Price', 'Marketing_Spend', 'Average_Customer_Spending', 'Reviews']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_features = ['Cuisine_Type', 'Promotions']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get the input values from the form
    Number_of_Customers = float(request.form["Number_of_Customers"])
    Menu_Price = float(request.form["Menu_Price"])
    Marketing_Spend = float(request.form["Marketing_Spend"])
    Cuisine_Type = request.form["Cuisine_Type"]
    Average_Customer_Spending = float(request.form["Average_Customer_Spending"])
    Promotions = float(request.form["Promotions"])
    Reviews = float(request.form["Reviews"])
    
    # Create a DataFrame with the input data
    input_data = pd.DataFrame({
        'Number_of_Customers': [Number_of_Customers],
        'Menu_Price': [Menu_Price],
        'Marketing_Spend': [Marketing_Spend],
        'Cuisine_Type': [Cuisine_Type],
        'Average_Customer_Spending': [Average_Customer_Spending],
        'Promotions': [Promotions],
        'Reviews': [Reviews]
    })
    
    # Fit the preprocessor with the training data
    X_train = pd.read_csv('data/Restaurant_revenue.csv')
    y_train = X_train.pop('Monthly_Revenue')
    preprocessor.fit(X_train)
    
    # Preprocess the input data
    preprocessed_data = preprocessor.transform(input_data)
    
    # Make prediction using the loaded model
    prediction = model.predict(preprocessed_data)
    
    # Format the prediction as a string
    predicted_revenue = f"${prediction[0]:,.2f}"
    
    # Render the result template with the prediction
    return render_template("result.html", prediction=predicted_revenue)

if __name__ == "__main__":
    app.run(debug=True)
