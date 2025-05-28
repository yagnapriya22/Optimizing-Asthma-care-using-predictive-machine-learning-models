import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from flask import Flask, request, render_template
import os

# 1. Data Collection - Creating synthetic dataset
np.random.seed(42)
n_samples = 1000

data = {
    'age': np.random.randint(5, 80, n_samples),
    'gender': np.random.choice([0, 1], n_samples),  # 0: Female, 1: Male
    'bmi': np.random.uniform(15, 40, n_samples),
    'smoking': np.random.choice([0, 1], n_samples),  # 0: No, 1: Yes
    'pet_allergy': np.random.choice([0, 1], n_samples),
    'dust_allergy': np.random.choice([0, 1], n_samples),
    'pollen_allergy': np.random.choice([0, 1], n_samples),
    'air_pollution_exposure': np.random.uniform(0, 100, n_samples),
    'exercise_frequency': np.random.randint(0, 7, n_samples),
    'family_history': np.random.choice([0, 1], n_samples),
    'respiratory_infections': np.random.randint(0, 5, n_samples),
    'wheezing': np.random.choice([0, 1], n_samples),
    'shortness_breath': np.random.choice([0, 1], n_samples),
    'chest_tightness': np.random.choice([0, 1], n_samples),
    'coughing': np.random.choice([0, 1], n_samples)
}

# Generate asthma_risk (0: Low, 1: Medium, 2: High) based on features
df = pd.DataFrame(data)
df['asthma_risk'] = (
    0.3 * df['wheezing'] +
    0.3 * df['shortness_breath'] +
    0.2 * df['chest_tightness'] +
    0.2 * df['coughing'] +
    0.1 * df['family_history'] +
    0.05 * df['air_pollution_exposure'] / 100 +
    0.05 * df['dust_allergy'] +
    0.05 * df['pollen_allergy']
)

# Handle NaN values and categorize
df['asthma_risk'] = pd.cut(
    df['asthma_risk'],
    bins=[-np.inf, 0.3, 0.6, np.inf],
    labels=[0, 1, 2],
    include_lowest=True
).astype(int)

# Save dataset
df.to_csv('asthma_dataset.csv', index=False)

# 2. Data Pre-processing
# Check for missing values
print("Missing values:\n", df.isnull().sum())

# Split features and target
X = df.drop('asthma_risk', axis=1)
y = df['asthma_risk']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Save scaler
joblib.dump(scaler, 'scaler.pkl')

# 3. Feature Selection
selector = SelectKBest(score_func=f_classif, k=10)
X_selected = selector.fit_transform(X_scaled, y)
selected_features = X.columns[selector.get_support()].tolist()
print("Selected features:", selected_features)

# Visualization: Feature Importance
feature_scores = pd.DataFrame({'Feature': X.columns, 'Score': selector.scores_})
plt.figure(figsize=(10, 6))
sns.barplot(x='Score', y='Feature', data=feature_scores.sort_values('Score', ascending=False))
plt.title('Feature Importance Scores')
plt.savefig('feature_importance.png')
plt.close()

# 4. Model Selection
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='rbf', random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'XGBoost': XGBClassifier(random_state=42, eval_metric='mlogloss')
}

# 5. Model Training and Evaluation
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
model_results = {}

for name, model in models.items():
    # Training
    model.fit(X_train, y_train)
    
    # Evaluation
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    model_results[name] = {
        'model': model,
        'accuracy': accuracy,
        'report': classification_report(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    
    # Save model
    joblib.dump(model, f'{name.lower().replace(" ", "_")}_model.pkl')
    
    # Visualization: Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(model_results[name]['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(f'confusion_matrix_{name.lower().replace(" ", "_")}.png')
    plt.close()

# 6. Model Comparison
# Visualization: Model Comparison
accuracies = [model_results[name]['accuracy'] for name in models]
plt.figure(figsize=(10, 6))
sns.barplot(x=accuracies, y=list(models.keys()))
plt.title('Model Accuracy Comparison')
plt.xlabel('Accuracy')
plt.savefig('model_comparison.png')
plt.close()

# Print model results
for name, result in model_results.items():
    print(f"\n{name} Results:")
    print(f"Accuracy: {result['accuracy']:.4f}")
    print("Classification Report:\n", result['report'])

# 7. Prediction Function
def predict_asthma(input_data, model, scaler, selected_features):
    input_df = pd.DataFrame([input_data], columns=X.columns)
    input_scaled = scaler.transform(input_df)
    input_selected = input_scaled[:, [X.columns.get_loc(f) for f in selected_features]]
    prediction = model.predict(input_selected)[0]
    
    risk_levels = {0: 'Low', 1: 'Medium', 2: 'High'}
    explanations = {
        0: "Low risk of asthma based on provided symptoms and environmental factors.",
        1: "Moderate risk of asthma. Some symptoms and risk factors are present.",
        2: "High risk of asthma. Multiple symptoms and strong risk factors detected."
    }
    suggestions = {
        0: "Maintain healthy lifestyle, avoid known allergens, and monitor for symptoms.",
        1: "Consult a healthcare provider, avoid triggers, and consider allergy testing.",
        2: "Seek immediate medical attention, avoid all triggers, and follow prescribed treatment."
    }
    
    return {
        'risk_level': risk_levels[prediction],
        'explanation': explanations[prediction],
        'suggestions': suggestions[prediction]
    }

# 8. Flask App for Front-end
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_data = {
        'age': float(request.form['age']),
        'gender': int(request.form['gender']),
        'bmi': float(request.form['bmi']),
        'smoking': int(request.form['smoking']),
        'pet_allergy': int(request.form['pet_allergy']),
        'dust_allergy': int(request.form['dust_allergy']),
        'pollen_allergy': int(request.form['pollen_allergy']),
        'air_pollution_exposure': float(request.form['air_pollution_exposure']),
        'exercise_frequency': int(request.form['exercise_frequency']),
        'family_history': int(request.form['family_history']),
        'respiratory_infections': int(request.form['respiratory_infections']),
        'wheezing': int(request.form['wheezing']),
        'shortness_breath': int(request.form['shortness_breath']),
        'chest_tightness': int(request.form['chest_tightness']),
        'coughing': int(request.form['coughing'])
    }
    
    # Load best model 
    model = joblib.load('random_forest_model.pkl')
    scaler = joblib.load('scaler.pkl')
    result = predict_asthma(input_data, model, scaler, selected_features)
    
    return render_template('result.html', result=result)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))  # Default to 5001 to avoid conflicts
    app.run(debug=True, port=port)