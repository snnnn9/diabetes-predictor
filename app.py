import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import io

# Set page config
st.set_page_config(
    page_title="Diabetes Predictor",
    page_icon="🩺",
    layout="wide"
)

# Define the neural network model
class DiabetesNet(nn.Module):
    def __init__(self, input_size):
        super(DiabetesNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.sigmoid(self.fc4(x))
        return x

# Function to train the model
@st.cache_data
def train_model(X_train, y_train, X_test, y_test):
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train.values.reshape(-1, 1))
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test.values.reshape(-1, 1))
    
    # Initialize model
    model = DiabetesNet(X_train.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    model.train()
    for epoch in range(100):  # Keep epochs low for speed
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
    
    # Evaluate model
    model.eval()
    with torch.no_grad():
        train_pred = model(X_train_tensor)
        test_pred = model(X_test_tensor)
        
        train_accuracy = accuracy_score(y_train, (train_pred.numpy() > 0.5).astype(int))
        test_accuracy = accuracy_score(y_test, (test_pred.numpy() > 0.5).astype(int))
    
    return model, train_accuracy, test_accuracy

# Function to make prediction
def make_prediction(model, scaler, input_data):
    # Scale the input data
    input_scaled = scaler.transform([input_data])
    
    # Convert to tensor and make prediction
    input_tensor = torch.FloatTensor(input_scaled)
    
    model.eval()
    with torch.no_grad():
        prediction = model(input_tensor)
        probability = prediction.item()
        result = 1 if probability > 0.5 else 0
    
    return result, probability

# Main Streamlit app
def main():
    st.title("🩺 Diabetes Predictor")
    st.markdown("---")
    
    # File upload section
    st.subheader("📊 Dataset Upload")
    uploaded_file = st.file_uploader("Upload Pima Indians Diabetes Dataset (CSV)", type=['csv'])
    
    if uploaded_file is not None:
        # Load the dataset
        try:
            df = pd.read_csv(uploaded_file)
            
            # Validate required columns
            required_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                              'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
            
            if not all(col in df.columns for col in required_columns):
                st.error("Dataset must contain all required columns: " + ", ".join(required_columns))
                return
            
            st.success(f"Dataset loaded successfully! Shape: {df.shape}")
            
            # Display basic statistics
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Dataset Preview:**")
                st.dataframe(df.head())
            
            with col2:
                st.write("**Target Distribution:**")
                outcome_counts = df['Outcome'].value_counts()
                st.write(f"Non-diabetic (0): {outcome_counts[0]}")
                st.write(f"Diabetic (1): {outcome_counts[1]}")
            
            # Prepare data for training
            feature_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                             'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
            
            X = df[feature_columns]
            y = df['Outcome']
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale the features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train the model
            st.subheader("🤖 Model Training")
            with st.spinner("Training neural network..."):
                model, train_acc, test_acc = train_model(X_train_scaled, y_train, X_test_scaled, y_test)
            
            st.success("Model trained successfully!")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Training Accuracy", f"{train_acc:.3f}")
            with col2:
                st.metric("Test Accuracy", f"{test_acc:.3f}")
            
            # User input section
            st.subheader("🔮 Make Prediction")
            st.markdown("Enter your health parameters in the sidebar to get a diabetes prediction.")
            
            # Sidebar for user input
            st.sidebar.header("📝 Enter Your Health Parameters")
            
            # Input fields with reasonable default values and ranges
            pregnancies = st.sidebar.slider("Pregnancies", 0, 17, 1)
            glucose = st.sidebar.slider("Glucose Level", 0, 200, 120)
            blood_pressure = st.sidebar.slider("Blood Pressure", 0, 122, 70)
            skin_thickness = st.sidebar.slider("Skin Thickness", 0, 99, 20)
            insulin = st.sidebar.slider("Insulin", 0, 846, 80)
            bmi = st.sidebar.slider("BMI", 0.0, 67.1, 25.0, 0.1)
            diabetes_pedigree = st.sidebar.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5, 0.01)
            age = st.sidebar.slider("Age", 21, 81, 30)
            
            # Make prediction button
            if st.sidebar.button("🔍 Predict Diabetes Risk", type="primary"):
                # Prepare input data
                input_data = [pregnancies, glucose, blood_pressure, skin_thickness, 
                            insulin, bmi, diabetes_pedigree, age]
                
                # Make prediction
                prediction, probability = make_prediction(model, scaler, input_data)
                
                # Display results
                st.markdown("---")
                st.subheader("📋 Prediction Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if prediction == 1:
                        st.error("⚠️ **HIGH RISK**")
                        st.write("The model predicts you may be at risk for diabetes.")
                    else:
                        st.success("✅ **LOW RISK**")
                        st.write("The model predicts you are likely not diabetic.")
                
                with col2:
                    st.metric("Prediction Probability", f"{probability:.3f}")
                    st.write(f"Confidence: {max(probability, 1-probability):.1%}")
                
                with col3:
                    st.info("**Disclaimer**")
                    st.write("This is a prediction model for educational purposes. Always consult healthcare professionals for medical advice.")
                
                # Display input summary
                st.subheader("📊 Your Input Summary")
                input_df = pd.DataFrame({
                    'Parameter': feature_columns,
                    'Your Value': input_data
                })
                st.dataframe(input_df, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error processing the dataset: {str(e)}")
    
    else:
        st.info("👆 Please upload the Pima Indians Diabetes dataset (CSV file) to begin.")
        st.markdown("""
        **Expected CSV format:**
        - Pregnancies
        - Glucose  
        - BloodPressure
        - SkinThickness
        - Insulin
        - BMI
        - DiabetesPedigreeFunction
        - Age
        - Outcome (0 = non-diabetic, 1 = diabetic)
        """)

if __name__ == "__main__":
    main()
