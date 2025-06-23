import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="🩺 Diabetes Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    
    .high-risk {
        background-color: #ffebee;
        border: 2px solid #f44336;
        color: #c62828;
    }
    
    .low-risk {
        background-color: #e8f5e8;
        border: 2px solid #4caf50;
        color: #2e7d32;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 20px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: transform 0.2s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 10px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# Neural Network Model Definition
class DiabetesPredictor(nn.Module):
    def __init__(self, input_size=8):
        super(DiabetesPredictor, self).__init__()
        # Define network layers
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Forward pass through the network
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc4(x))
        return x

@st.cache_data
def load_and_preprocess_data():
    """Load and preprocess the diabetes dataset"""
    try:
        # Load the dataset
        df = pd.read_csv('diabetes.csv')
        
        # Feature columns
        feature_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                          'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        
        # Separate features and target
        X = df[feature_columns].values
        y = df['Outcome'].values
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        return df, X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_columns
        
    except FileNotFoundError:
        st.error("⚠️ diabetes.csv file not found! Please ensure the file is in the same directory as this app.")
        return None, None, None, None, None, None, None

@st.cache_resource
def train_model(X_train, y_train, X_test, y_test):
    """Train the PyTorch model"""
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test).reshape(-1, 1)
    
    # Initialize the model
    model = DiabetesPredictor()
    criterion = nn.BCELoss()  # Binary Cross Entropy Loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    model.train()
    epochs = 1000
    for epoch in range(epochs):
        # Forward pass
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Evaluate the model
    model.eval()
    with torch.no_grad():
        train_predictions = model(X_train_tensor)
        test_predictions = model(X_test_tensor)
        
        # Convert to binary predictions
        train_pred_binary = (train_predictions > 0.5).float()
        test_pred_binary = (test_predictions > 0.5).float()
        
        # Calculate accuracy
        train_accuracy = accuracy_score(y_train, train_pred_binary.numpy())
        test_accuracy = accuracy_score(y_test, test_pred_binary.numpy())
    
    return model, train_accuracy, test_accuracy

def make_prediction(model, scaler, user_input):
    """Make a prediction using the trained model"""
    # Scale the input
    user_input_scaled = scaler.transform([user_input])
    
    # Convert to tensor
    input_tensor = torch.FloatTensor(user_input_scaled)
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        prediction = model(input_tensor)
        probability = prediction.item()
    
    return probability

def create_feature_distributions(df):
    """Create visualizations of feature distributions"""
    # Create subplots
    fig = make_subplots(
        rows=2, cols=4,
        subplot_titles=['Pregnancies', 'Glucose', 'Blood Pressure', 'Skin Thickness',
                       'Insulin', 'BMI', 'Diabetes Pedigree', 'Age'],
        specs=[[{"secondary_y": False}]*4]*2
    )
    
    features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
               'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    
    positions = [(1,1), (1,2), (1,3), (1,4), (2,1), (2,2), (2,3), (2,4)]
    
    for i, (feature, pos) in enumerate(zip(features, positions)):
        # Separate data by outcome
        diabetic = df[df['Outcome'] == 1][feature]
        non_diabetic = df[df['Outcome'] == 0][feature]
        
        # Add histograms
        fig.add_trace(
            go.Histogram(x=non_diabetic, name='Non-Diabetic', opacity=0.7, 
                        marker_color='lightblue', showlegend=(i==0)),
            row=pos[0], col=pos[1]
        )
        fig.add_trace(
            go.Histogram(x=diabetic, name='Diabetic', opacity=0.7, 
                        marker_color='lightcoral', showlegend=(i==0)),
            row=pos[0], col=pos[1]
        )
    
    fig.update_layout(height=600, title_text="Feature Distributions by Diabetes Outcome")
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">🩺 Diabetes Predictor</h1>', unsafe_allow_html=True)
    st.markdown("### AI-Powered Diabetes Risk Assessment using Deep Learning")
    
    # Load and preprocess data
    data = load_and_preprocess_data()
    if data[0] is None:
        return
    
    df, X_train, X_test, y_train, y_test, scaler, feature_columns = data
    
    # Train model
    with st.spinner("🧠 Training the AI model... This may take a moment."):
        model, train_acc, test_acc = train_model(X_train, y_train, X_test, y_test)
    
    # Sidebar for user input
    st.sidebar.header("📋 Patient Information")
    st.sidebar.markdown("Please enter the patient's medical information:")
    
    # User input form
    with st.sidebar.form("prediction_form"):
        pregnancies = st.number_input("👶 Number of Pregnancies", min_value=0, max_value=20, value=1, step=1)
        glucose = st.number_input("🍯 Glucose Level (mg/dL)", min_value=0, max_value=300, value=120, step=1)
        blood_pressure = st.number_input("💓 Blood Pressure (mmHg)", min_value=0, max_value=200, value=80, step=1)
        skin_thickness = st.number_input("📏 Skin Thickness (mm)", min_value=0, max_value=100, value=20, step=1)
        insulin = st.number_input("💉 Insulin Level (μU/mL)", min_value=0, max_value=900, value=80, step=1)
        bmi = st.number_input("⚖️ BMI (Body Mass Index)", min_value=0.0, max_value=70.0, value=25.0, step=0.1)
        diabetes_pedigree = st.number_input("🧬 Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.01)
        age = st.number_input("🎂 Age (years)", min_value=1, max_value=120, value=30, step=1)
        
        predict_button = st.form_submit_button("🔮 Predict Diabetes Risk")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Dataset overview
        st.subheader("📊 Dataset Overview")
        
        # Basic statistics
        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            st.metric("Total Samples", len(df))
        with col_b:
            st.metric("Diabetic Cases", df['Outcome'].sum())
        with col_c:
            st.metric("Non-Diabetic Cases", len(df) - df['Outcome'].sum())
        with col_d:
            st.metric("Diabetes Rate", f"{(df['Outcome'].mean()*100):.1f}%")
        
        # Feature distributions
        st.subheader("📈 Feature Distributions")
        distribution_fig = create_feature_distributions(df)
        st.plotly_chart(distribution_fig, use_container_width=True)
        
        # Model performance
        st.subheader("🎯 Model Performance")
        perf_col1, perf_col2 = st.columns(2)
        with perf_col1:
            st.metric("Training Accuracy", f"{train_acc:.3f}")
        with perf_col2:
            st.metric("Testing Accuracy", f"{test_acc:.3f}")
    
    with col2:
        st.subheader("🔍 Prediction Results")
        
        if predict_button:
            # Prepare user input
            user_input = [pregnancies, glucose, blood_pressure, skin_thickness, 
                         insulin, bmi, diabetes_pedigree, age]
            
            # Make prediction
            probability = make_prediction(model, scaler, user_input)
            
            # Display results
            if probability > 0.5:
                st.markdown(f'''
                <div class="prediction-box high-risk">
                    ⚠️ HIGH RISK<br>
                    Probability: {probability:.1%}
                </div>
                ''', unsafe_allow_html=True)
                st.warning("⚠️ **Recommendation:** Please consult with a healthcare professional for further evaluation and testing.")
            else:
                st.markdown(f'''
                <div class="prediction-box low-risk">
                    ✅ LOW RISK<br>
                    Probability: {probability:.1%}
                </div>
                ''', unsafe_allow_html=True)
                st.success("✅ **Good news!** The model indicates a low risk of diabetes. Continue maintaining a healthy lifestyle.")
            
            # Risk factors analysis
            st.subheader("📋 Risk Factors Analysis")
            
            # Define normal ranges (simplified)
            risk_factors = []
            if glucose > 140:
                risk_factors.append("High glucose level")
            if blood_pressure > 90:
                risk_factors.append("High blood pressure")
            if bmi > 30:
                risk_factors.append("High BMI (obesity)")
            if age > 45:
                risk_factors.append("Advanced age")
            
            if risk_factors:
                st.warning("⚠️ **Identified Risk Factors:**")
                for factor in risk_factors:
                    st.write(f"• {factor}")
            else:
                st.success("✅ No major risk factors identified.")
        
        else:
            st.info("👈 Please fill in the patient information in the sidebar and click 'Predict' to get the diabetes risk assessment.")
        
        # Educational content
        st.subheader("📚 About Diabetes")
        st.markdown("""
        **Type 2 Diabetes** is a chronic condition that affects how your body processes blood sugar (glucose).
        
        **Key Risk Factors:**
        - Age (45+ years)
        - Obesity (BMI > 30)
        - Family history
        - High blood pressure
        - High glucose levels
        
        **Prevention Tips:**
        - Maintain healthy weight
        - Exercise regularly
        - Eat a balanced diet
        - Monitor blood sugar
        - Regular check-ups
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <small>
        ⚠️ <strong>Disclaimer:</strong> This tool is for educational purposes only and should not replace professional medical advice. 
        Always consult with a healthcare provider for accurate diagnosis and treatment.
        </small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
