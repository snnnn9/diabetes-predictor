import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go

# Set page config
st.set_page_config(
    page_title="Diabetes Predictor",
    page_icon="🩺",
    layout="wide"
)

# Define the neural network model
class DiabetesNet(nn.Module):
    def __init__(self, input_size=8):
        super(DiabetesNet, self).__init__()
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

@st.cache_data
def load_and_preprocess_data(uploaded_file):
    """Load and preprocess the diabetes dataset"""
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        # Create sample data if no file uploaded (for demo purposes)
        np.random.seed(42)
        n_samples = 500
        df = pd.DataFrame({
            'Pregnancies': np.random.randint(0, 15, n_samples),
            'Glucose': np.random.normal(120, 30, n_samples),
            'BloodPressure': np.random.normal(70, 15, n_samples),
            'SkinThickness': np.random.normal(25, 10, n_samples),
            'Insulin': np.random.normal(100, 50, n_samples),
            'BMI': np.random.normal(28, 5, n_samples),
            'DiabetesPedigreeFunction': np.random.uniform(0.1, 2.0, n_samples),
            'Age': np.random.randint(21, 80, n_samples),
            'Outcome': np.random.binomial(1, 0.35, n_samples)
        })
        # Ensure positive values where needed
        df['Glucose'] = np.clip(df['Glucose'], 50, 200)
        df['BloodPressure'] = np.clip(df['BloodPressure'], 40, 120)
        df['SkinThickness'] = np.clip(df['SkinThickness'], 5, 50)
        df['Insulin'] = np.clip(df['Insulin'], 10, 300)
        df['BMI'] = np.clip(df['BMI'], 15, 50)
    
    return df

@st.cache_resource
def train_model(X_train, y_train, X_test, y_test):
    """Train the PyTorch model"""
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test).view(-1, 1)
    
    # Initialize model
    model = DiabetesNet()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    epochs = 100
    train_losses = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        
        # Update progress
        progress = (epoch + 1) / epochs
        progress_bar.progress(progress)
        status_text.text(f'Training... Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')
    
    # Evaluate model
    model.eval()
    with torch.no_grad():
        train_pred = model(X_train_tensor)
        test_pred = model(X_test_tensor)
        
        train_pred_binary = (train_pred > 0.5).float()
        test_pred_binary = (test_pred > 0.5).float()
        
        train_acc = accuracy_score(y_train, train_pred_binary.numpy())
        test_acc = accuracy_score(y_test, test_pred_binary.numpy())
    
    progress_bar.empty()
    status_text.empty()
    
    return model, train_losses, train_acc, test_acc

def main():
    st.title("🩺 Diabetes Predictor")
    st.markdown("### Predict diabetes risk using machine learning")
    
    # Sidebar for file upload and user inputs
    st.sidebar.header("📊 Data & Prediction")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "Upload Pima Indians Diabetes CSV", 
        type=['csv'],
        help="Upload the diabetes dataset CSV file"
    )
    
    if uploaded_file is None:
        st.sidebar.info("💡 No file uploaded. Using demo data for training.")
    
    # Load data
    df = load_and_preprocess_data(uploaded_file)
    
    # Display dataset info
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Dataset Overview")
        st.write(f"**Samples:** {len(df)}")
        st.write(f"**Features:** {len(df.columns)-1}")
        st.write(f"**Diabetic cases:** {df['Outcome'].sum()} ({df['Outcome'].mean():.1%})")
        
        # Show dataset statistics
        st.write("**Dataset Statistics:**")
        st.dataframe(df.describe(), use_container_width=True)
    
    with col2:
        st.subheader("🎯 Outcome Distribution")
        outcome_counts = df['Outcome'].value_counts()
        fig = px.pie(
            values=outcome_counts.values, 
            names=['Non-Diabetic', 'Diabetic'],
            color_discrete_sequence=['#2E8B57', '#DC143C']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Prepare data for training
    feature_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                      'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    
    X = df[feature_columns].values
    y = df['Outcome'].values
    
    # Split and scale data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model button
    if st.button("🚀 Train Model", type="primary"):
        st.subheader("🤖 Model Training")
        
        with st.spinner("Training neural network..."):
            model, train_losses, train_acc, test_acc = train_model(
                X_train_scaled, y_train, X_test_scaled, y_test
            )
        
        # Store model and scaler in session state
        st.session_state.model = model
        st.session_state.scaler = scaler
        st.session_state.trained = True
        
        # Display training results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Training Accuracy", f"{train_acc:.2%}")
        with col2:
            st.metric("Test Accuracy", f"{test_acc:.2%}")
        with col3:
            st.metric("Final Loss", f"{train_losses[-1]:.4f}")
        
        # Plot training loss
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(len(train_losses))),
            y=train_losses,
            mode='lines',
            name='Training Loss',
            line=dict(color='#1f77b4')
        ))
        fig.update_layout(
            title="Training Loss Over Time",
            xaxis_title="Epoch",
            yaxis_title="Loss",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.success("✅ Model trained successfully!")
    
    # Prediction section
    if hasattr(st.session_state, 'trained') and st.session_state.trained:
        st.sidebar.markdown("---")
        st.sidebar.subheader("🔮 Make Prediction")
        
        # Input fields for prediction
        pregnancies = st.sidebar.slider("Pregnancies", 0, 17, 1)
        glucose = st.sidebar.slider("Glucose Level", 50, 200, 120)
        blood_pressure = st.sidebar.slider("Blood Pressure", 40, 120, 70)
        skin_thickness = st.sidebar.slider("Skin Thickness", 5, 50, 25)
        insulin = st.sidebar.slider("Insulin", 10, 300, 100)
        bmi = st.sidebar.slider("BMI", 15.0, 50.0, 28.0, 0.1)
        diabetes_pedigree = st.sidebar.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5, 0.01)
        age = st.sidebar.slider("Age", 21, 80, 35)
        
        # Make prediction
        if st.sidebar.button("🎯 Predict", type="primary"):
            # Prepare input data
            input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                                  insulin, bmi, diabetes_pedigree, age]])
            
            # Scale input data
            input_scaled = st.session_state.scaler.transform(input_data)
            input_tensor = torch.FloatTensor(input_scaled)
            
            # Make prediction
            st.session_state.model.eval()
            with torch.no_grad():
                prediction = st.session_state.model(input_tensor)
                probability = prediction.item()
            
            # Display prediction
            st.subheader("🎯 Prediction Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if probability > 0.5:
                    st.error(f"⚠️ **High Diabetes Risk**")
                    st.write(f"Probability: **{probability:.1%}**")
                else:
                    st.success(f"✅ **Low Diabetes Risk**")
                    st.write(f"Probability: **{probability:.1%}**")
            
            with col2:
                # Create probability gauge
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = probability * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Diabetes Risk %"},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgreen"},
                            {'range': [50, 80], 'color': "yellow"},
                            {'range': [80, 100], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            # Show input summary
            st.subheader("📋 Input Summary")
            input_df = pd.DataFrame({
                'Feature': feature_columns,
                'Value': [pregnancies, glucose, blood_pressure, skin_thickness,
                         insulin, bmi, diabetes_pedigree, age]
            })
            st.dataframe(input_df, use_container_width=True)
    
    else:
        st.sidebar.info("👆 Please train the model first to make predictions")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**Note:** This is a demonstration app for educational purposes. "
        "Always consult healthcare professionals for medical advice."
    )

if __name__ == "__main__":
    main()
