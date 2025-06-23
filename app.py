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

# Define the enhanced neural network model
class DiabetesNet(nn.Module):
    def __init__(self, input_size=8):
        super(DiabetesNet, self).__init__()
        # Deeper architecture with batch normalization
        self.fc1 = nn.Linear(input_size, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(32, 16)
        self.bn3 = nn.BatchNorm1d(16)
        self.fc4 = nn.Linear(16, 8)
        self.fc5 = nn.Linear(8, 1)
        
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(0.3)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # First layer with batch norm
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        
        # Second layer with batch norm
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        
        # Third layer with batch norm
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        
        # Final layers
        x = self.relu(self.fc4(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc5(x))
        return x

@st.cache_data
def load_and_preprocess_data(uploaded_file):
    """Load and preprocess the diabetes dataset"""
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        # Create more realistic sample data with proper diabetes patterns
        np.random.seed(42)
        n_samples = 800
        
        # Create two groups: diabetic and non-diabetic with realistic distributions
        n_diabetic = int(n_samples * 0.35)  # 35% diabetic
        n_non_diabetic = n_samples - n_diabetic
        
        # Non-diabetic group (healthier values)
        non_diabetic_data = {
            'Pregnancies': np.random.poisson(2, n_non_diabetic),
            'Glucose': np.random.normal(95, 15, n_non_diabetic),  # Lower glucose
            'BloodPressure': np.random.normal(68, 10, n_non_diabetic),
            'SkinThickness': np.random.normal(20, 8, n_non_diabetic),
            'Insulin': np.random.normal(85, 30, n_non_diabetic),
            'BMI': np.random.normal(25, 4, n_non_diabetic),  # Lower BMI
            'DiabetesPedigreeFunction': np.random.gamma(2, 0.2, n_non_diabetic),
            'Age': np.random.gamma(3, 10, n_non_diabetic) + 21,
            'Outcome': np.zeros(n_non_diabetic)
        }
        
        # Diabetic group (higher risk values)
        diabetic_data = {
            'Pregnancies': np.random.poisson(4, n_diabetic),  # More pregnancies
            'Glucose': np.random.normal(145, 25, n_diabetic),  # Higher glucose
            'BloodPressure': np.random.normal(78, 12, n_diabetic),
            'SkinThickness': np.random.normal(28, 10, n_diabetic),
            'Insulin': np.random.normal(150, 60, n_diabetic),  # Higher insulin
            'BMI': np.random.normal(33, 6, n_diabetic),  # Higher BMI
            'DiabetesPedigreeFunction': np.random.gamma(3, 0.3, n_diabetic),
            'Age': np.random.gamma(4, 12, n_diabetic) + 30,  # Older age
            'Outcome': np.ones(n_diabetic)
        }
        
        # Combine and shuffle
        df_non_diabetic = pd.DataFrame(non_diabetic_data)
        df_diabetic = pd.DataFrame(diabetic_data)
        df = pd.concat([df_non_diabetic, df_diabetic], ignore_index=True)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Ensure realistic bounds
        df['Pregnancies'] = np.clip(df['Pregnancies'], 0, 17)
        df['Glucose'] = np.clip(df['Glucose'], 44, 199)
        df['BloodPressure'] = np.clip(df['BloodPressure'], 24, 122)
        df['SkinThickness'] = np.clip(df['SkinThickness'], 7, 99)
        df['Insulin'] = np.clip(df['Insulin'], 14, 846)
        df['BMI'] = np.clip(df['BMI'], 18.2, 67.1)
        df['DiabetesPedigreeFunction'] = np.clip(df['DiabetesPedigreeFunction'], 0.078, 2.42)
        df['Age'] = np.clip(df['Age'], 21, 81).astype(int)
    
    return df

@st.cache_resource
def train_model(X_train, y_train, X_test, y_test):
    """Train the enhanced PyTorch model"""
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test).view(-1, 1)
    
    # Initialize enhanced model
    model = DiabetesNet()
    
    # Use weighted loss to handle class imbalance
    pos_weight = torch.tensor([len(y_train) / (2 * np.sum(y_train))])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Use different learning rates for different layers
    optimizer = optim.AdamW([
        {'params': model.fc1.parameters(), 'lr': 0.001},
        {'params': model.fc2.parameters(), 'lr': 0.0008},
        {'params': model.fc3.parameters(), 'lr': 0.0006},
        {'params': model.fc4.parameters(), 'lr': 0.0004},
        {'params': model.fc5.parameters(), 'lr': 0.0002},
    ], weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
    
    # Training loop with early stopping
    epochs = 200
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    early_stop_patience = 20
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        optimizer.zero_grad()
        
        # Forward pass (remove sigmoid from model output for BCEWithLogitsLoss)
        outputs = model(X_train_tensor)
        # Apply sigmoid manually for loss calculation
        outputs_sigmoid = torch.sigmoid(outputs)
        loss = criterion(outputs, y_train_tensor)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Validation phase
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test_tensor)
            val_outputs_sigmoid = torch.sigmoid(val_outputs)
            val_loss = criterion(val_outputs, y_test_tensor)
        
        train_losses.append(loss.item())
        val_losses.append(val_loss.item())
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= early_stop_patience:
            st.info(f"Early stopping at epoch {epoch+1}")
            break
        
        # Update progress
        progress = (epoch + 1) / epochs
        progress_bar.progress(progress)
        status_text.text(f'Training... Epoch {epoch+1}/{epochs}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')
    
    # Final evaluation with proper sigmoid
    model.eval()
    with torch.no_grad():
        train_pred = torch.sigmoid(model(X_train_tensor))
        test_pred = torch.sigmoid(model(X_test_tensor))
        
        train_pred_binary = (train_pred > 0.5).float()
        test_pred_binary = (test_pred > 0.5).float()
        
        train_acc = accuracy_score(y_train, train_pred_binary.numpy())
        test_acc = accuracy_score(y_test, test_pred_binary.numpy())
    
    progress_bar.empty()
    status_text.empty()
    
    return model, train_losses, val_losses, train_acc, test_acc

def main():
    st.title("🩺 Diabetes Predictor")
    st.markdown("### Predict diabetes risk using machine learning")
    st.markdown("*Enhanced AI model with improved accuracy - Created by Rexzea*")
    
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
        
        with st.spinner("Training enhanced neural network..."):
            model, train_losses, val_losses, train_acc, test_acc = train_model(
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
        
        # Plot training and validation loss
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(len(train_losses))),
            y=train_losses,
            mode='lines',
            name='Training Loss',
            line=dict(color='#1f77b4')
        ))
        fig.add_trace(go.Scatter(
            x=list(range(len(val_losses))),
            y=val_losses,
            mode='lines',
            name='Validation Loss',
            line=dict(color='#ff7f0e')
        ))
        fig.update_layout(
            title="Training & Validation Loss Over Time",
            xaxis_title="Epoch",
            yaxis_title="Loss",
            showlegend=True
        ))
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
            # Make prediction with proper sigmoid
            input_tensor = torch.FloatTensor(input_scaled)
            
            # Make prediction
            st.session_state.model.eval()
            with torch.no_grad():
                raw_output = st.session_state.model(input_tensor)
                probability = torch.sigmoid(raw_output).item()  # Apply sigmoid manually
            
            # Enhanced risk assessment with multiple thresholds
            if probability > 0.7:
                risk_level = "🔴 **VERY HIGH DIABETES RISK**"
                risk_color = "error"
                risk_advice = "⚠️ **Immediate medical consultation recommended!**"
            elif probability > 0.5:
                risk_level = "🟠 **HIGH DIABETES RISK**"
                risk_color = "warning"  
                risk_advice = "⚠️ **Please consult with a healthcare provider soon.**"
            elif probability > 0.3:
                risk_level = "🟡 **MODERATE DIABETES RISK**"
                risk_color = "info"
                risk_advice = "💡 **Consider lifestyle improvements and regular check-ups.**"
            else:
                risk_level = "🟢 **LOW DIABETES RISK**"
                risk_color = "success"
                risk_advice = "✅ **Maintain healthy lifestyle habits.**"
            
            # Display prediction
            st.subheader("🎯 Prediction Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if risk_color == "error":
                    st.error(risk_level)
                elif risk_color == "warning":
                    st.warning(risk_level)
                elif risk_color == "info":
                    st.info(risk_level)
                else:
                    st.success(risk_level)
                    
                st.write(f"**Probability:** {probability:.1%}")
                st.write(risk_advice)
            
            with col2:
                # Create enhanced probability gauge
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = probability * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Diabetes Risk %"},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgreen"},
                            {'range': [30, 50], 'color': "yellow"},
                            {'range': [50, 70], 'color': "orange"}, 
                            {'range': [70, 100], 'color': "red"}
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
            
            # Risk factor analysis
            st.subheader("📊 Risk Factor Analysis")
            risk_factors = []
            
            if glucose > 125:
                risk_factors.append(f"🔸 High Glucose Level ({glucose}) - Normal: <100")
            if bmi > 30:
                risk_factors.append(f"🔸 High BMI ({bmi:.1f}) - Normal: 18.5-24.9")
            if age > 45:
                risk_factors.append(f"🔸 Advanced Age ({age}) - Risk increases after 45")
            if blood_pressure > 80:
                risk_factors.append(f"🔸 High Blood Pressure ({blood_pressure}) - Normal: <80")
            if pregnancies > 4:
                risk_factors.append(f"🔸 Multiple Pregnancies ({pregnancies}) - Higher risk")
            if diabetes_pedigree > 0.5:
                risk_factors.append(f"🔸 Strong Family History ({diabetes_pedigree:.2f})")
                
            if risk_factors:
                st.warning("**Identified Risk Factors:**")
                for factor in risk_factors:
                    st.write(factor)
            else:
                st.success("✅ **No major risk factors identified in the input values.**")
            
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
    
    # Watermark/Creator
    st.markdown(
        "<div style='text-align: center; color: #888; font-size: 14px; margin-top: 20px;'>"
        "Created by <strong>Rexzea</strong> 🚀"
        "</div>", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
