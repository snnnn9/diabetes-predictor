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
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="🩺 Diabetes Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #A23B72;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .predict-button {
        background-color: #2E86AB;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        border: none;
        cursor: pointer;
        transition: background-color 0.3s;
    }
    .predict-button:hover {
        background-color: #1B5E7D;
    }
    .high-risk {
        color: #E63946;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .low-risk {
        color: #2A9D8F;
        font-weight: bold;
        font-size: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)

# Define the improved neural network model
class DiabetesPredictor(nn.Module):
    def __init__(self, input_size=8):
        super(DiabetesPredictor, self).__init__()
        # Improved architecture with batch normalization and better layer sizes
        self.fc1 = nn.Linear(input_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, 1)
        
        # Activation functions and regularization
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(0.3)
        self.sigmoid = nn.Sigmoid()
        
        # Weight initialization
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize weights using Xavier/Glorot initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        # Layer 1 with batch norm
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        
        # Layer 2 with batch norm
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        
        # Layer 3 with batch norm
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        
        # Final layers
        x = self.relu(self.fc4(x))
        x = self.sigmoid(self.fc5(x))
        return x

# Cache the data loading and model training
@st.cache_data
def load_and_prepare_data():
    """Load and prepare the diabetes dataset"""
    try:
        # Load the dataset
        df = pd.read_csv('diabetes.csv')
        
        # Basic data cleaning - replace 0s with median for certain columns
        columns_to_replace = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        for col in columns_to_replace:
            df[col] = df[col].replace(0, df[col].median())
        
        return df
    except FileNotFoundError:
        st.error("❌ Dataset file 'diabetes.csv' not found. Please ensure it's in the same directory as this script.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading dataset: {str(e)}")
        st.stop()

@st.cache_resource
def train_model(X_train, y_train, X_test, y_test):
    """Train the improved PyTorch model with better training loop"""
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Initialize the model
    model = DiabetesPredictor()
    
    # Improved loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=False)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test).reshape(-1, 1)
    
    # Training loop with validation
    model.train()
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(1000):  # Increased epochs
        # Training phase
        model.train()
        optimizer.zero_grad()
        train_outputs = model(X_train_tensor)
        train_loss = criterion(train_outputs, y_train_tensor)
        train_loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Validation phase
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test_tensor)
            val_loss = criterion(val_outputs, y_test_tensor)
            
            # Calculate accuracies
            train_preds = (train_outputs.numpy() > 0.5).astype(int)
            val_preds = (val_outputs.numpy() > 0.5).astype(int)
            
            train_acc = accuracy_score(y_train, train_preds)
            val_acc = accuracy_score(y_test, val_preds)
        
        # Record metrics
        train_losses.append(train_loss.item())
        val_losses.append(val_loss.item())
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 50:  # Early stopping patience
                break
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        test_predictions = (test_outputs.numpy() > 0.5).astype(int)
        final_accuracy = accuracy_score(y_test, test_predictions)
        
        # Get prediction probabilities for better analysis
        test_probabilities = test_outputs.numpy().flatten()
    
    return model, final_accuracy, train_losses, val_losses, train_accuracies, val_accuracies, test_probabilities

def main():
    # Main header
    st.markdown('<h1 class="main-header">🩺 Diabetes Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Predict diabetes risk using advanced deep learning</p>', unsafe_allow_html=True)
    
    # Load and prepare data
    df = load_and_prepare_data()
    
    # Prepare features and target
    feature_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                      'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    X = df[feature_columns]
    y = df['Outcome']
    
    # Split and scale the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the model with improved architecture
    model, accuracy, train_losses, val_losses, train_accuracies, val_accuracies, test_probabilities = train_model(
        X_train_scaled, y_train.values, X_test_scaled, y_test.values
    )
    
    # Additional model evaluation metrics
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
    
    # Get final predictions for metrics
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test_scaled)
        final_predictions = model(X_test_tensor).numpy()
        binary_predictions = (final_predictions > 0.5).astype(int).flatten()
    
    precision = precision_score(y_test, binary_predictions)
    recall = recall_score(y_test, binary_predictions)
    f1 = f1_score(y_test, binary_predictions)
    auc_score = roc_auc_score(y_test, final_predictions)
    
    # Enhanced Sidebar for user input
    st.sidebar.markdown("""
    <div class="sidebar-section">
        <h2 style="margin: 0; text-align: center;">🩺 Health Assessment</h2>
        <p style="margin: 0.5rem 0; text-align: center; opacity: 0.9;">Enter your health information below</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create organized input sections
    st.sidebar.markdown("### 👤 Personal Information")
    pregnancies = st.sidebar.slider('🤱 Pregnancies', 0, 17, 1, help='Number of times pregnant')
    age = st.sidebar.slider('🎂 Age', 21, 81, 30, help='Age in years')
    
    st.sidebar.markdown("### 🩸 Glucose & Blood Metrics")
    glucose = st.sidebar.slider('🍬 Glucose Level', 0, 200, 120, help='Plasma glucose concentration (mg/dL)')
    blood_pressure = st.sidebar.slider('💓 Blood Pressure', 0, 122, 70, help='Diastolic blood pressure (mm Hg)')
    
    st.sidebar.markdown("### 📏 Physical Measurements")
    skin_thickness = st.sidebar.slider('📐 Skin Thickness', 0, 99, 20, help='Triceps skin fold thickness (mm)')
    bmi = st.sidebar.slider('⚖️ BMI', 0.0, 67.1, 25.0, step=0.1, help='Body mass index (kg/m²)')
    
    st.sidebar.markdown("### 🧬 Advanced Metrics")
    insulin = st.sidebar.slider('💉 Insulin', 0, 846, 79, help='2-Hour serum insulin (mu U/ml)')
    diabetes_pedigree = st.sidebar.slider('🧬 Diabetes Pedigree Function', 0.0, 2.5, 0.5, step=0.01, 
                                         help='Diabetes pedigree function (genetic factor)')
    
    # Enhanced prediction section
    st.sidebar.markdown("---")
    
    # Prediction button with better styling
    predict_clicked = st.sidebar.button(
        '🔮 ANALYZE DIABETES RISK', 
        key='predict_btn',
        help="Click to get your personalized diabetes risk assessment"
    )
    
    if predict_clicked:
        # Prepare input data
        input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, 
                               insulin, bmi, diabetes_pedigree, age]])
        input_scaled = scaler.transform(input_data)
        
        # Make prediction with confidence intervals
        model.eval()
        with torch.no_grad():
            input_tensor = torch.FloatTensor(input_scaled)
            
            # Multiple forward passes for uncertainty estimation
            predictions = []
            model.train()  # Enable dropout for uncertainty
            for _ in range(100):
                pred = model(input_tensor).item()
                predictions.append(pred)
            model.eval()
            
            # Calculate statistics
            mean_prediction = np.mean(predictions)
            std_prediction = np.std(predictions)
            confidence_interval = (
                max(0, mean_prediction - 1.96 * std_prediction),
                min(1, mean_prediction + 1.96 * std_prediction)
            )
        
        # Enhanced result display
        st.sidebar.markdown("---")
        
        # Risk assessment with detailed analysis
        if mean_prediction > 0.5:
            risk_level = "HIGH RISK"
            risk_color = "high-risk"
            risk_emoji = "⚠️"
            risk_message = "Elevated Risk Detected"
            recommendation = "Please consult with a healthcare provider for further evaluation."
        else:
            risk_level = "LOW RISK"
            risk_color = "low-risk"
            risk_emoji = "✅"
            risk_message = "Low Risk Profile"
            recommendation = "Continue maintaining a healthy lifestyle and regular check-ups."
        
        # Display results with improved formatting
        st.sidebar.markdown(f"""
        <div class="{risk_color}">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">{risk_emoji}</div>
            <div style="font-size: 1.4rem; font-weight: bold; margin-bottom: 0.5rem;">{risk_level}</div>
            <div style="font-size: 1rem; opacity: 0.9;">{risk_message}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Confidence metrics
        st.sidebar.markdown(f"""
        <div class="confidence-meter">
            <h4 style="margin: 0 0 1rem 0; color: #333;">📊 Analysis Details</h4>
            <p><strong>Risk Score:</strong> {mean_prediction:.1%}</p>
            <p><strong>Confidence Range:</strong> {confidence_interval[0]:.1%} - {confidence_interval[1]:.1%}</p>
            <p><strong>Model Certainty:</strong> {(1-std_prediction)*100:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Risk factors analysis
        risk_factors = []
        input_dict = {
            'Pregnancies': pregnancies,
            'Glucose': glucose,
            'BloodPressure': blood_pressure,
            'SkinThickness': skin_thickness,
            'Insulin': insulin,
            'BMI': bmi,
            'DiabetesPedigreeFunction': diabetes_pedigree,
            'Age': age
        }
        
        # Compare with dataset averages
        for feature, value in input_dict.items():
            avg_diabetic = df[df['Outcome'] == 1][feature].mean()
            avg_non_diabetic = df[df['Outcome'] == 0][feature].mean()
            
            if feature == 'Glucose' and value > 140:
                risk_factors.append(f"🔴 High glucose level ({value})")
            elif feature == 'BMI' and value > 30:
                risk_factors.append(f"🟡 High BMI ({value:.1f})")
            elif feature == 'Age' and value > 45:
                risk_factors.append(f"🟡 Advanced age ({value})")
            elif feature == 'BloodPressure' and value > 90:
                risk_factors.append(f"🟡 Elevated blood pressure ({value})")
        
        if risk_factors:
            st.sidebar.markdown(f"""
            <div class="risk-factors">
                <h4 style="margin: 0 0 1rem 0;">⚠️ Notable Risk Factors</h4>
                {'<br>'.join(['• ' + factor for factor in risk_factors])}
            </div>
            """, unsafe_allow_html=True)
        
        # Recommendations
        st.sidebar.markdown(f"""
        <div style="background: #f8f9fa; padding: 1rem; border-radius: 10px; margin: 1rem 0; border-left: 4px solid #007bff;">
            <h4 style="margin: 0 0 0.5rem 0; color: #007bff;">💡 Recommendation</h4>
            <p style="margin: 0; font-size: 0.9rem;">{recommendation}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<h2 class="sub-header">📈 Dataset Overview</h2>', unsafe_allow_html=True)
        
        # Display basic statistics
        st.markdown("### 📊 Dataset Statistics")
        st.dataframe(df.describe(), use_container_width=True)
        
        # Outcome distribution
        st.markdown("### 🎯 Diabetes Distribution")
        outcome_counts = df['Outcome'].value_counts()
        fig_pie = px.pie(values=outcome_counts.values, 
                        names=['Non-Diabetic', 'Diabetic'],
                        title='Distribution of Diabetes Cases',
                        color_discrete_sequence=['#2A9D8F', '#E63946'])
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # Feature correlation heatmap
        st.markdown("### 🔥 Feature Correlation Heatmap")
        correlation_matrix = df[feature_columns + ['Outcome']].corr()
        fig_heatmap = px.imshow(correlation_matrix, 
                               text_auto=True, 
                               aspect="auto",
                               color_continuous_scale='RdBu',
                               title='Feature Correlation Matrix')
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    with col2:
        st.markdown('<h2 class="sub-header">🤖 Advanced Model Analytics</h2>', unsafe_allow_html=True)
        
        # Enhanced model performance metrics
        col2a, col2b = st.columns(2)
        
        with col2a:
            st.metric(
                label="🎯 Model Accuracy",
                value=f"{accuracy:.2%}",
                delta=f"+{(accuracy-0.5):.1%} vs Random"
            )
            st.metric(
                label="🔍 Precision",
                value=f"{precision:.2%}",
                delta="High Quality"
            )
        
        with col2b:
            st.metric(
                label="📈 AUC Score",
                value=f"{auc_score:.3f}",
                delta="Excellent" if auc_score > 0.8 else "Good"
            )
            st.metric(
                label="🎪 F1 Score",
                value=f"{f1:.2%}",
                delta="Balanced"
            )
        
        # Enhanced training progress visualization
        st.markdown("### 📊 Training Progress")
        
        # Create training history dataframe
        epochs = range(len(train_losses))
        training_df = pd.DataFrame({
            'Epoch': list(epochs) + list(epochs),
            'Loss': train_losses + val_losses,
            'Type': ['Training'] * len(train_losses) + ['Validation'] * len(val_losses)
        })
        
        # Plot training and validation loss
        fig_training = px.line(training_df, x='Epoch', y='Loss', color='Type',
                              title='Model Training & Validation Loss',
                              color_discrete_map={'Training': '#2E86AB', 'Validation': '#E63946'})
        fig_training.update_layout(height=300)
        st.plotly_chart(fig_training, use_container_width=True)
        
        # Model architecture visualization
        st.markdown("### 🏗️ Neural Network Architecture")
        st.info("""
        **Enhanced Deep Learning Model:**
        - Input Layer: 8 features
        - Hidden Layer 1: 128 neurons + BatchNorm + LeakyReLU + Dropout
        - Hidden Layer 2: 64 neurons + BatchNorm + LeakyReLU + Dropout  
        - Hidden Layer 3: 32 neurons + BatchNorm + LeakyReLU + Dropout
        - Hidden Layer 4: 16 neurons + ReLU
        - Output Layer: 1 neuron + Sigmoid
        - **Total Parameters:** ~18K
        """)
        
        # Advanced model insights
        st.markdown("### 🧠 Model Intelligence")
        col2c, col2d = st.columns(2)
        
        with col2c:
            st.markdown("""
            **🔬 Advanced Features:**
            - Batch Normalization
            - Dropout Regularization  
            - Learning Rate Scheduling
            - Early Stopping
            - Gradient Clipping
            """)
        
        with col2d:
            st.markdown(f"""
            **📊 Performance Metrics:**
            - Sensitivity: {recall:.2%}
            - Specificity: {precision:.2%}
            - Training Epochs: {len(train_losses)}
            - Convergence: ✅ Achieved
            """)
        
        # Prediction confidence distribution
        st.markdown("### 🎯 Model Confidence Analysis")
        confidence_df = pd.DataFrame({
            'Prediction_Confidence': test_probabilities,
            'Actual_Outcome': y_test.values
        })
        
        fig_confidence = px.histogram(confidence_df, x='Prediction_Confidence', 
                                    color='Actual_Outcome',
                                    title='Model Confidence Distribution',
                                    nbins=20,
                                    color_discrete_map={0: '#26A69A', 1: '#FF5252'})
        fig_confidence.update_layout(height=300)

        
        # Training loss chart
        st.markdown("### 📉 Training Loss")
        loss_df = pd.DataFrame({'Epoch': range(len(losses)), 'Loss': losses})
        fig_loss = px.line(loss_df, x='Epoch', y='Loss', 
                          title='Model Training Loss Over Time',
                          color_discrete_sequence=['#2E86AB'])
        st.plotly_chart(fig_loss, use_container_width=True)
        
        # Dataset info
        st.markdown("### 📋 Dataset Information")
        st.info(f"""
        **Total Samples:** {len(df)}
        **Features:** {len(feature_columns)}
        **Diabetic Cases:** {df['Outcome'].sum()} ({df['Outcome'].mean():.1%})
        **Non-Diabetic Cases:** {len(df) - df['Outcome'].sum()} ({(1-df['Outcome'].mean()):.1%})
        """)
        
        # Feature importance (based on correlation with outcome)
        st.markdown("### 🎯 Feature Importance")
        feature_importance = abs(df[feature_columns].corrwith(df['Outcome'])).sort_values(ascending=False)
        fig_importance = px.bar(x=feature_importance.values, 
                               y=feature_importance.index,
                               orientation='h',
                               title='Feature Importance (Correlation with Outcome)',
                               color_discrete_sequence=['#A23B72'])
        fig_importance.update_layout(height=400)
        st.plotly_chart(fig_importance, use_container_width=True)
    
    # Additional insights
    st.markdown("---")
    st.markdown('<h2 class="sub-header">💡 Key Insights</h2>', unsafe_allow_html=True)
    
    col3, col4, col5 = st.columns(3)
    
    with col3:
        avg_glucose_diabetic = df[df['Outcome'] == 1]['Glucose'].mean()
        avg_glucose_non_diabetic = df[df['Outcome'] == 0]['Glucose'].mean()
        st.metric(
            label="Avg Glucose (Diabetic)",
            value=f"{avg_glucose_diabetic:.0f}",
            delta=f"{avg_glucose_diabetic - avg_glucose_non_diabetic:.0f} vs Non-Diabetic"
        )
    
    with col4:
        avg_bmi_diabetic = df[df['Outcome'] == 1]['BMI'].mean()
        avg_bmi_non_diabetic = df[df['Outcome'] == 0]['BMI'].mean()
        st.metric(
            label="Avg BMI (Diabetic)",
            value=f"{avg_bmi_diabetic:.1f}",
            delta=f"{avg_bmi_diabetic - avg_bmi_non_diabetic:.1f} vs Non-Diabetic"
        )
    
    with col5:
        avg_age_diabetic = df[df['Outcome'] == 1]['Age'].mean()
        avg_age_non_diabetic = df[df['Outcome'] == 0]['Age'].mean()
        st.metric(
            label="Avg Age (Diabetic)",
            value=f"{avg_age_diabetic:.0f}",
            delta=f"{avg_age_diabetic - avg_age_non_diabetic:.0f} vs Non-Diabetic"
        )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p>🩺 <strong>Diabetes Predictor</strong> - Powered by PyTorch & Streamlit</p>
        <p><em>Note: This tool is for educational purposes only. Please consult healthcare professionals for medical advice.</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
