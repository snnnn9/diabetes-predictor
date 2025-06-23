import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import io

# Set page config
st.set_page_config(
    page_title="Diabetes Predictor",
    page_icon="🩺",
    layout="wide"
)

# Define the Neural Network Model
class DiabetesPredictor(nn.Module):
    def __init__(self, input_size=8):
        super(DiabetesPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc3(x))
        return x

# Function to train the model
@st.cache_data
def train_model(data):
    """Train the PyTorch model on the diabetes dataset"""
    
    # Prepare features and target
    feature_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                      'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    X = data[feature_columns].values
    y = data['Outcome'].values
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_test_tensor = torch.FloatTensor(y_test).reshape(-1, 1)
    
    # Initialize the model
    model = DiabetesPredictor()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    model.train()
    losses = []
    for epoch in range(100):  # Keep epochs low for speed
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    
    # Evaluate the model
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        test_predictions = (test_outputs > 0.5).float()
        accuracy = accuracy_score(y_test, test_predictions.numpy())
    
    return model, scaler, accuracy, losses

# Function to make predictions
def make_prediction(model, scaler, features):
    """Make a prediction using the trained model"""
    features_scaled = scaler.transform([features])
    features_tensor = torch.FloatTensor(features_scaled)
    
    model.eval()
    with torch.no_grad():
        prediction = model(features_tensor)
        probability = prediction.item()
        result = 1 if probability > 0.5 else 0
    
    return result, probability

# Main Streamlit App
def main():
    st.title("🩺 Diabetes Predictor")
    st.markdown("### Predict diabetes risk using machine learning")
    st.markdown("---")
    
    # File uploader for dataset
    uploaded_file = st.file_uploader(
        "Upload the Pima Indians Diabetes dataset (CSV file)",
        type=['csv'],
        help="Upload a CSV file containing the diabetes dataset"
    )
    
    if uploaded_file is not None:
        try:
            # Load the dataset
            data = pd.read_csv(uploaded_file)
            
            # Validate required columns
            required_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                              'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
            
            if not all(col in data.columns for col in required_columns):
                st.error("❌ Dataset must contain all required columns: " + ", ".join(required_columns))
                return
            
            # Display dataset info
            st.success("✅ Dataset loaded successfully!")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Records", len(data))
                st.metric("Diabetic Cases", sum(data['Outcome'] == 1))
            with col2:
                st.metric("Non-diabetic Cases", sum(data['Outcome'] == 0))
                st.metric("Features", len(required_columns) - 1)
            
            # Show first few rows
            with st.expander("📊 View Dataset Sample"):
                st.dataframe(data.head())
            
            # Train the model
            with st.spinner("🧠 Training the neural network model..."):
                model, scaler, accuracy, losses = train_model(data)
            
            st.success(f"✅ Model trained successfully! Accuracy: {accuracy:.2%}")
            
            # Show training progress
            with st.expander("📈 Training Progress"):
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(losses)
                ax.set_title("Training Loss Over Time")
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Loss")
                st.pyplot(fig)
            
            st.markdown("---")
            
            # User input section
            st.subheader("🔍 Make a Prediction")
            st.markdown("Enter your health information below:")
            
            # Create input form in sidebar
            with st.sidebar:
                st.header("📝 Input Features")
                
                pregnancies = st.number_input(
                    "Pregnancies", 
                    min_value=0, max_value=20, value=1,
                    help="Number of times pregnant"
                )
                
                glucose = st.number_input(
                    "Glucose Level", 
                    min_value=0, max_value=200, value=120,
                    help="Plasma glucose concentration (mg/dL)"
                )
                
                blood_pressure = st.number_input(
                    "Blood Pressure", 
                    min_value=0, max_value=150, value=80,
                    help="Diastolic blood pressure (mm Hg)"
                )
                
                skin_thickness = st.number_input(
                    "Skin Thickness", 
                    min_value=0, max_value=100, value=20,
                    help="Triceps skin fold thickness (mm)"
                )
                
                insulin = st.number_input(
                    "Insulin Level", 
                    min_value=0, max_value=900, value=80,
                    help="2-Hour serum insulin (mu U/ml)"
                )
                
                bmi = st.number_input(
                    "BMI", 
                    min_value=10.0, max_value=70.0, value=25.0, step=0.1,
                    help="Body mass index (weight in kg/(height in m)^2)"
                )
                
                diabetes_pedigree = st.number_input(
                    "Diabetes Pedigree Function", 
                    min_value=0.0, max_value=3.0, value=0.5, step=0.01,
                    help="Diabetes pedigree function (genetic factor)"
                )
                
                age = st.number_input(
                    "Age", 
                    min_value=18, max_value=100, value=30,
                    help="Age in years"
                )
                
                predict_button = st.button("🔮 Predict", type="primary")
            
            # Make prediction when button is clicked
            if predict_button:
                features = [pregnancies, glucose, blood_pressure, skin_thickness, 
                          insulin, bmi, diabetes_pedigree, age]
                
                result, probability = make_prediction(model, scaler, features)
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    if result == 1:
                        st.error("⚠️ **HIGH RISK** - Diabetes Detected")
                        st.write(f"**Confidence:** {probability:.1%}")
                    else:
                        st.success("✅ **LOW RISK** - No Diabetes Detected")
                        st.write(f"**Confidence:** {(1-probability):.1%}")
                
                with col2:
                    # Show input summary
                    st.info("📋 **Input Summary**")
                    st.write(f"• Pregnancies: {pregnancies}")
                    st.write(f"• Glucose: {glucose} mg/dL")
                    st.write(f"• Blood Pressure: {blood_pressure} mm Hg")
                    st.write(f"• BMI: {bmi}")
                    st.write(f"• Age: {age} years")
                
                # Health recommendations
                st.markdown("---")
                st.subheader("💡 Health Recommendations")
                
                if result == 1:
                    st.warning("""
                    **Important Notice:** This prediction indicates a higher risk of diabetes. 
                    Please consider:
                    - Consulting with a healthcare professional
                    - Regular monitoring of blood glucose levels
                    - Maintaining a healthy diet and exercise routine
                    - Managing stress levels
                    """)
                else:
                    st.info("""
                    **Great News:** Your risk appears to be lower, but remember:
                    - Continue maintaining a healthy lifestyle
                    - Regular health check-ups are still important
                    - Monitor your weight and exercise regularly
                    - Keep a balanced diet
                    """)
        
        except Exception as e:
            st.error(f"❌ Error processing the dataset: {str(e)}")
    
    else:
        st.info("👆 Please upload the Pima Indians Diabetes dataset CSV file to get started.")
        
        # Show sample data format
        st.subheader("📋 Expected Data Format")
        sample_data = {
            'Pregnancies': [6, 1, 8],
            'Glucose': [148, 85, 183],
            'BloodPressure': [72, 66, 64],
            'SkinThickness': [35, 29, 0],
            'Insulin': [0, 0, 0],
            'BMI': [33.6, 26.6, 23.3],
            'DiabetesPedigreeFunction': [0.627, 0.351, 0.672],
            'Age': [50, 31, 32],
            'Outcome': [1, 0, 1]
        }
        st.dataframe(pd.DataFrame(sample_data))

# Disclaimer
    st.markdown("---")
    st.markdown("""
    **⚠️ Medical Disclaimer:** This application is for educational and informational purposes only. 
    It should not be used as a substitute for professional medical advice, diagnosis, or treatment. 
    Always consult with qualified healthcare providers for medical decisions.
    """)

if __name__ == "__main__":
    main()
