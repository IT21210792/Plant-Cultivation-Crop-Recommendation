from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    BaggingClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, confusion_matrix
from skopt import BayesSearchCV  # Bayesian Optimization

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

# Load the dataset
try:
    crop = pd.read_csv("Crop_recommendation.csv")
except Exception as e:
    print(f"Error loading CSV: {e}")
    crop = pd.DataFrame()

# Validate numeric columns
numeric_columns = ['N (kg/ha)', 'P (kg/ha)', 'K (kg/ha)', 'Temp (°C)', 'Humidity (%)', 'pH', 'Rainfall (mm)']
for col in numeric_columns:
    if col in crop.columns:
        crop[col] = pd.to_numeric(crop[col], errors='coerce')
        if crop[col].isna().any():
            print(f"Warning: Column {col} contains invalid values, filling with median.")
            crop[col].fillna(crop[col].median(), inplace=True)
    else:
        print(f"Error: Column {col} not found in CSV.")
        crop[col] = 0  # Add missing column with default value

# Map labels to numerical values
crop_dict = {
    'ginger': 1, 'turmeric': 2, 'cinnamon': 3, 'coffee': 4
}
reverse_crop_dict = {v: k for k, v in crop_dict.items()}

if 'label' in crop.columns:
    crop['label'] = crop['label'].map(crop_dict)
    if crop['label'].isna().any():
        print("Warning: Some labels could not be mapped, dropping invalid rows.")
        crop = crop.dropna(subset=['label'])
else:
    print("Error: 'label' column not found in CSV.")
    crop['label'] = 1  # Default to ginger

# Create guide lookup dictionary
guide_lookup = {}
if 'guide' in crop.columns and 'label' in crop.columns:
    guide_lookup = dict(zip(crop['label'].map(reverse_crop_dict), crop['guide']))
else:
    print("Error: 'guide' or 'label' column missing, guide lookup will be empty.")
    guide_lookup = {crop_name: "No cultivation guide available." for crop_name in crop_dict.keys()}

# Split data into features and labels
X = crop[numeric_columns]
y = crop['label']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply MinMaxScaler
mx = MinMaxScaler()
X_train = mx.fit_transform(X_train)
X_test = mx.transform(X_test)

# Apply StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Initialize models
models = {
    'LogisticRegression': LogisticRegression(),
    'GaussianNB': GaussianNB(),
    'SVC': SVC(),
    'KNeighborsClassifier': KNeighborsClassifier(),
    'DecisionTreeClassifier': DecisionTreeClassifier(),
    'ExtraTreeClassifier': ExtraTreeClassifier(),
    'RandomForestClassifier': RandomForestClassifier(),
    'BaggingClassifier': BaggingClassifier(),
    'GradientBoostingClassifier': GradientBoostingClassifier(),
    'AdaBoostClassifier': AdaBoostClassifier(),
}

# Evaluate models
best_model_name = ""
best_accuracy = 0.0

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    
    print(f"{name}:")
    print(f"Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}, Precision: {precision:.4f}\n")

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model_name = name

print(f"\nBest Model: {best_model_name} with accuracy: {best_accuracy:.4f}")

# Train the best model
best_model = models[best_model_name]

# Apply Bayesian Optimization only if the model supports hyperparameter tuning
if best_model_name == "GaussianNB":
    print("Skipping Bayesian Optimization because GaussianNB does not require hyperparameter tuning.")
else:
    param_space = {}
    
    if best_model_name == "RandomForestClassifier":
        param_space = {'n_estimators': (10, 200), 'max_depth': (5, 50)}
    elif best_model_name == "SVC":
        param_space = {'C': (0.1, 10), 'gamma': (0.01, 1)}
    elif best_model_name == "KNeighborsClassifier":
        param_space = {'n_neighbors': (1, 20)}
    elif best_model_name == "GradientBoostingClassifier":
        param_space = {'n_estimators': (50, 300), 'learning_rate': (0.01, 0.2)}

    if param_space:
        opt = BayesSearchCV(best_model, param_space, n_iter=10, cv=3, scoring='accuracy', random_state=42)
        opt.fit(X_train, y_train)
        best_model = opt.best_estimator_
        print(f"Optimized {best_model_name} with Bayesian Optimization.")

# Save the model and scalers
pickle.dump(best_model, open('model.pkl', 'wb'))
pickle.dump(mx, open('minmaxscaler.pkl', 'wb'))
pickle.dump(sc, open('standscaler.pkl', 'wb'))

# Function for crop recommendation
def recommendation(N, P, K, temperature, humidity, ph, rainfall):
    input_data = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]], 
                              columns=numeric_columns)
    mx_features = mx.transform(input_data)
    sc_mx_features = sc.transform(mx_features)
    prediction = best_model.predict(sc_mx_features)
    return prediction[0]

@app.route('/process', methods=['POST'])
def process():
    try:
        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])
        
        # Predict crop and get the crop name
        predicted_crop = recommendation(N, P, K, temperature, humidity, ph, rainfall)
        predicted_crop_name = reverse_crop_dict.get(predicted_crop, "Unknown Crop")

        # Lookup guide text
        guide_text = guide_lookup.get(predicted_crop_name, "No cultivation guide available.")

        # Generate Confusion Matrix and display heatmap
        y_pred_test = best_model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred_test)

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix for {best_model_name}")
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.savefig('static/confusion_matrix.png')  # Save confusion matrix as an image
        plt.close()

        print(f"Confusion Matrix for {best_model_name}:")
        print(cm)

        return render_template('index.html',
                              label=predicted_crop_name,  # Output crop name
                              guide=guide_text,           # Output guide text
                              matrix_img='static/confusion_matrix.png')
    except ValueError as e:
        error_message = "Error: Invalid input. Please enter numeric values for all fields."
        return render_template('index.html', label="Error", guide=error_message)



if __name__ == '__main__':
    import os
    os.environ['FLASK_RUN_EXTRA_FILES'] = ''  # Prevent Flask from monitoring extra files
    app.run(debug=True, use_reloader=False)  # Disable auto-restart
