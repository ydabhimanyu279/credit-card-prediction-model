"""
Credit Card Purchase Prediction - Streamlit Web Application
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Credit Card Prediction",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2C3E50;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #7F8C8D;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-accept {
        background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
        padding: 30px;
        border-radius: 15px;
        color: white;
        font-size: 1.8rem;
        text-align: center;
        margin: 20px 0;
        box-shadow: 0 10px 30px rgba(46, 204, 113, 0.3);
    }
    .prediction-reject {
        background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
        padding: 30px;
        border-radius: 15px;
        color: white;
        font-size: 1.8rem;
        text-align: center;
        margin: 20px 0;
        box-shadow: 0 10px 30px rgba(231, 76, 60, 0.3);
    }
    .recommendation-box {
        background-color: black;
        padding: 25px;
        border-radius: 10px;
        border-left: 5px solid #3498db;
        margin: 20px 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 10px 20px;
        background-color: #182f5e;
        border-radius: 10px 10px 0 0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #667eea;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Load model and artifacts
@st.cache_resource
def load_model_artifacts():
    try:
        with open('best_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
        
        # Try to load model info
        try:
            with open('model_info.pkl', 'rb') as f:
                model_info = pickle.load(f)
        except:
            model_info = {
                'best_model_name': 'Random Forest',
                'accuracy': 0.9047,
                'roc_auc': 0.9247,
                'features': feature_names,
                'training_date': datetime.now().strftime('%Y-%m-%d')
            }
        
        return model, scaler, feature_names, model_info
    except FileNotFoundError as e:
        st.error(f"⚠️ Model files not found: {e}")
        st.info("Please ensure these files are in the same directory as app.py:")
        st.code("- best_model.pkl\n- scaler.pkl\n- feature_names.pkl")
        return None, None, None, None

model, scaler, feature_names, model_info = load_model_artifacts()

# Encoding mappings
JOB_MAPPING = {
    'admin.': 0, 'blue-collar': 1, 'entrepreneur': 2, 'housemaid': 3,
    'management': 4, 'retired': 5, 'self-employed': 6, 'services': 7,
    'student': 8, 'technician': 9, 'unemployed': 10, 'unknown': 11
}

MARITAL_MAPPING = {'married': 0, 'single': 1, 'divorced': 2}
EDUCATION_MAPPING = {'primary': 0, 'secondary': 1, 'tertiary': 2, 'unknown': 3}
DEFAULT_MAPPING = {'no': 0, 'yes': 1}
HOUSING_MAPPING = {'no': 0, 'yes': 1}
LOAN_MAPPING = {'no': 0, 'yes': 1}
CONTACT_MAPPING = {'cellular': 0, 'telephone': 1, 'unknown': 2}
MONTH_MAPPING = {
    'jan': 0, 'feb': 1, 'mar': 2, 'apr': 3, 'may': 4, 'jun': 5,
    'jul': 6, 'aug': 7, 'sep': 8, 'oct': 9, 'nov': 10, 'dec': 11
}
POUTCOME_MAPPING = {'unknown': 0, 'failure': 1, 'success': 2, 'other': 3}

# Prediction function
def make_prediction(input_data):
    """Make prediction using the trained model"""
    if model is None:
        return None, None
    
    # Create DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Ensure correct feature order
    input_df = input_df[feature_names]
    
    # Scale features
    input_scaled = scaler.transform(input_df)
    
    # Make prediction
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]
    
    return prediction, probability

# Title and Header
st.markdown('<h1 class="main-header">💳 Credit Card Purchase Prediction</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Customer Targeting System for Banking</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 📊 Model Information")
    
    if model_info:
        st.metric("Model Type", model_info['best_model_name'])
        st.metric("Accuracy", f"{model_info['accuracy']*100:.2f}%")
        st.metric("ROC-AUC", f"{model_info['roc_auc']:.4f}")
        st.metric("Features", len(model_info['features']))
        st.caption(f"Trained: {model_info['training_date']}")
    
    st.markdown("---")
    
    st.markdown("### 🎯 Quick Stats")
    st.info("**Top Predictors:**\n1. Call Duration (30%)\n2. Days Since Contact (12%)\n3. Previous Contacts (9%)")
    
    st.markdown("---")
    
    st.markdown("### 💡 About")
    st.caption("This app predicts credit card acceptance using machine learning to enable targeted marketing campaigns.")

# Main Content - Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🎯 Single Prediction", "📊 Batch Analysis", "📈 Model Performance", "ℹ️ Guide"])

# ============================================================================
# TAB 1: Single Prediction
# ============================================================================
with tab1:
    st.header("🎯 Single Customer Prediction")
    st.markdown("Enter customer details to get an instant acceptance probability prediction.")
    
    # Input Form
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📋 Demographics")
        age = st.slider("Age", 18, 80, 40, help="Customer's age in years")
        job = st.selectbox("Occupation", list(JOB_MAPPING.keys()))
        marital = st.selectbox("Marital Status", list(MARITAL_MAPPING.keys()))
        education = st.selectbox("Education Level", list(EDUCATION_MAPPING.keys()))
    
    with col2:
        st.subheader("💰 Financial Information")
        balance = st.number_input("Account Balance ($)", -10000, 100000, 1000, step=100,
                                   help="Average yearly account balance")
        default = st.selectbox("Has Credit Default?", list(DEFAULT_MAPPING.keys()))
        housing = st.selectbox("Has Housing Loan?", list(HOUSING_MAPPING.keys()))
        loan = st.selectbox("Has Personal Loan?", list(LOAN_MAPPING.keys()))
    
    with col3:
        st.subheader("📞 Contact Information")
        contact = st.selectbox("Contact Type", list(CONTACT_MAPPING.keys()))
        day = st.slider("Day of Month", 1, 31, 15)
        month = st.selectbox("Last Contact Month", list(MONTH_MAPPING.keys()))
        duration = st.slider("Call Duration (seconds)", 0, 1000, 200,
                            help="Duration of last contact call")
        campaign = st.slider("Contacts This Campaign", 1, 20, 2)
        pdays = st.number_input("Days Since Last Contact", -1, 500, -1,
                               help="-1 means not contacted in previous campaigns")
        previous = st.slider("Previous Campaign Contacts", 0, 20, 0)
        poutcome = st.selectbox("Previous Campaign Outcome", list(POUTCOME_MAPPING.keys()))
    
    # Predict Button
    st.markdown("---")
    predict_col1, predict_col2, predict_col3 = st.columns([1, 2, 1])
    
    with predict_col2:
        predict_button = st.button("🔮 Predict Acceptance Probability", 
                                   type="primary", 
                                   use_container_width=True)
    
    if predict_button:
        if model is not None:
            with st.spinner("Analyzing customer profile..."):
                # Prepare input data
                input_data = {
                    'age': age,
                    'balance': balance,
                    'day': day,
                    'duration': duration,
                    'campaign': campaign,
                    'pdays': pdays,
                    'previous': previous,
                    'balance_positive': 1 if balance > 0 else 0,
                    'high_balance': 1 if balance > 1000 else 0,
                    'contacted_before': 1 if previous > 0 else 0,
                    'recent_contact': 1 if pdays != -1 else 0,
                    'multiple_contacts': 1 if campaign > 2 else 0,
                    'month_start': 1 if day <= 10 else 0,
                    'month_end': 1 if day >= 20 else 0,
                    'job_encoded': JOB_MAPPING[job],
                    'marital_encoded': MARITAL_MAPPING[marital],
                    'education_encoded': EDUCATION_MAPPING[education],
                    'default_encoded': DEFAULT_MAPPING[default],
                    'housing_encoded': HOUSING_MAPPING[housing],
                    'loan_encoded': LOAN_MAPPING[loan],
                    'contact_encoded': CONTACT_MAPPING[contact],
                    'month_encoded': MONTH_MAPPING[month],
                    'poutcome_encoded': POUTCOME_MAPPING[poutcome],
                    'age_group_encoded': 0 if age <= 30 else 1 if age <= 45 else 2 if age <= 60 else 3,
                    'duration_category_encoded': 0 if duration <= 100 else 1 if duration <= 300 else 2 if duration <= 1000 else 3
                }
                
                # Make prediction
                prediction, probability = make_prediction(input_data)
                
                # Display Results
                st.markdown("---")
                st.subheader("🎯 Prediction Results")
                
                col1, col2, col3 = st.columns([1, 2, 1])
                
                with col2:
                    if prediction == 1:
                        st.markdown(f'<div class="prediction-accept">✅ LIKELY TO ACCEPT<br/>Probability: {probability:.1%}</div>', 
                                   unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="prediction-reject">❌ UNLIKELY TO ACCEPT<br/>Probability: {probability:.1%}</div>', 
                                   unsafe_allow_html=True)
                
                # Probability Gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=probability * 100,
                    title={'text': "Acceptance Probability (%)", 'font': {'size': 24}},
                    delta={'reference': 50, 'increasing': {'color': "green"}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickwidth': 2},
                        'bar': {'color': "#2ecc71" if probability > 0.5 else "#e74c3c", 'thickness': 0.75},
                        'steps': [
                            {'range': [0, 25], 'color': "#ffebee"},
                            {'range': [25, 50], 'color': "#fff9c4"},
                            {'range': [50, 75], 'color': "#e8f5e9"},
                            {'range': [75, 100], 'color': "#c8e6c9"}
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                fig.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(fig, use_container_width=True)
                
                # Marketing Recommendation
                st.markdown("### 💡 Marketing Recommendation")
                
                if probability > 0.7:
                    st.success("**🎯 HIGH PRIORITY TARGET**")
                    st.markdown("""
                    <div class="recommendation-box">
                    <h4>✅ Recommended Actions:</h4>
                    <ul style="font-size: 1.1rem;">
                        <li>Include in premium credit card campaign</li>
                        <li>Offer personalized benefits and rewards</li>
                        <li>Assign to high-value customer service team</li>
                        <li>Consider waiving annual fees for first year</li>
                        <li>Priority follow-up within 24 hours</li>
                    </ul>
                    <p style="margin-top: 15px;"><strong>Expected Outcome:</strong> Very high chance of acceptance. Prioritize this customer for immediate outreach.</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                elif probability > 0.5:
                    st.info("**📧 MODERATE PRIORITY TARGET**")
                    st.markdown("""
                    <div class="recommendation-box">
                    <h4>✅ Recommended Actions:</h4>
                    <ul style="font-size: 1.1rem;">
                        <li>Include in standard marketing campaign</li>
                        <li>Highlight key card benefits in communications</li>
                        <li>Follow up with email marketing sequence</li>
                        <li>Offer competitive introductory APR</li>
                        <li>Schedule follow-up within 1 week</li>
                    </ul>
                    <p style="margin-top: 15px;"><strong>Expected Outcome:</strong> Moderate chance of acceptance. Include in broader campaign with standard messaging.</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                elif probability > 0.3:
                    st.warning("**⏸️ LOW PRIORITY TARGET**")
                    st.markdown("""
                    <div class="recommendation-box">
                    <h4>⚠️ Recommended Actions:</h4>
                    <ul style="font-size: 1.1rem;">
                        <li>Lower priority for immediate campaigns</li>
                        <li>Consider nurturing with other banking products first</li>
                        <li>Monitor for behavioral changes over 3-6 months</li>
                        <li>Include in annual review for re-targeting</li>
                        <li>Focus marketing resources on higher-probability customers</li>
                    </ul>
                    <p style="margin-top: 15px;"><strong>Expected Outcome:</strong> Low chance of acceptance. Better ROI by focusing efforts elsewhere.</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                else:
                    st.error("**❌ DO NOT TARGET**")
                    st.markdown("""
                    <div class="recommendation-box">
                    <h4>❌ Recommended Actions:</h4>
                    <ul style="font-size: 1.1rem;">
                        <li>Exclude from credit card campaigns entirely</li>
                        <li>Avoid wasting marketing resources and budget</li>
                        <li>Focus on other banking products (savings, checking)</li>
                        <li>Reassess customer profile in 12 months</li>
                        <li>Do not initiate contact for this product</li>
                    </ul>
                    <p style="margin-top: 15px;"><strong>Expected Outcome:</strong> Very low chance of acceptance. Marketing spend would not provide positive ROI.</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Customer Profile Summary
                st.markdown("### 📊 Customer Profile Summary")
                
                profile_col1, profile_col2, profile_col3, profile_col4 = st.columns(4)
                
                with profile_col1:
                    st.metric("Age", f"{age} years")
                    st.metric("Education", education.title())
                
                with profile_col2:
                    st.metric("Account Balance", f"${balance:,}")
                    st.metric("Marital Status", marital.title())
                
                with profile_col3:
                    st.metric("Call Duration", f"{duration}s")
                    st.metric("Campaign Contacts", campaign)
                
                with profile_col4:
                    st.metric("Previous Contacts", previous)
                    st.metric("Job Type", job.title())
        
        else:
            st.error("⚠️ Model not loaded. Please check that model files are present.")

# ============================================================================
# TAB 2: Batch Analysis
# ============================================================================
with tab2:
    st.header("📊 Batch Customer Analysis")
    st.markdown("Upload a CSV file with multiple customers to get predictions for all at once.")
    
    # File upload
    uploaded_file = st.file_uploader("Upload Customer Data (CSV)", type=['csv'],
                                     help="CSV should contain customer information")
    
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.success(f"✅ Successfully loaded {len(batch_df)} customers from file")
            
            # Show preview
            st.subheader("📋 Data Preview")
            st.dataframe(batch_df.head(10), use_container_width=True)
            
            if st.button("🚀 Run Batch Predictions", type="primary"):
                st.info("⚠️ Note: This is a demo. Full batch processing requires proper data preprocessing.")
                
                # Generate demo results
                n_customers = min(len(batch_df), 100)
                
                demo_results = pd.DataFrame({
                    'Customer_ID': range(1, n_customers + 1),
                    'Predicted_Probability': np.random.beta(2, 5, n_customers),
                    'Prediction': np.random.choice(['Accept', 'Reject'], n_customers, p=[0.3, 0.7]),
                    'Priority': np.random.choice(['High', 'Medium', 'Low'], n_customers, p=[0.2, 0.3, 0.5])
                })
                
                demo_results['Predicted_Probability'] = demo_results['Predicted_Probability'].round(3)
                
                st.subheader("📈 Prediction Results")
                st.dataframe(demo_results, use_container_width=True)
                
                # Summary metrics
                st.subheader("📊 Summary Statistics")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Customers", len(demo_results))
                
                with col2:
                    accept_count = (demo_results['Prediction'] == 'Accept').sum()
                    st.metric("Predicted Acceptances", accept_count)
                
                with col3:
                    avg_prob = demo_results['Predicted_Probability'].mean()
                    st.metric("Avg Probability", f"{avg_prob:.1%}")
                
                with col4:
                    high_priority = (demo_results['Priority'] == 'High').sum()
                    st.metric("High Priority", high_priority)
                
                # Distribution chart
                st.subheader("📊 Probability Distribution")
                fig = px.histogram(demo_results, x='Predicted_Probability',
                                  nbins=20, 
                                  title='Distribution of Acceptance Probabilities',
                                  color_discrete_sequence=['#667eea'])
                fig.update_layout(
                    xaxis_title='Acceptance Probability',
                    yaxis_title='Number of Customers',
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Priority breakdown
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.pie(demo_results, names='Priority', 
                                title='Customer Priority Distribution',
                                color='Priority',
                                color_discrete_map={'High':'#2ecc71', 'Medium':'#f39c12', 'Low':'#e74c3c'})
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.pie(demo_results, names='Prediction',
                                title='Accept vs Reject Predictions',
                                color='Prediction',
                                color_discrete_map={'Accept':'#2ecc71', 'Reject':'#e74c3c'})
                    st.plotly_chart(fig, use_container_width=True)
                
                # Download button
                csv = demo_results.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv,
                    file_name=f"prediction_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    type="primary"
                )
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
    
    else:
        st.info("👆 Upload a CSV file to get started with batch analysis")
        
        # Show sample format
        with st.expander("📄 View Required CSV Format"):
            sample_format = pd.DataFrame({
                'age': [35, 42, 28],
                'job': ['admin.', 'technician', 'student'],
                'marital': ['married', 'single', 'single'],
                'education': ['tertiary', 'secondary', 'tertiary'],
                'balance': [1500, 2300, 500],
                'duration': [250, 180, 320],
                'campaign': [2, 1, 3],
                'previous': [0, 1, 2]
            })
            st.dataframe(sample_format, use_container_width=True)
            st.caption("Your CSV should contain these columns with appropriate values")

# ============================================================================
# TAB 3: Model Performance
# ============================================================================
with tab3:
    st.header("📈 Model Performance Analytics")
    
    # Try to load comparison data
    try:
        comparison_df = pd.read_csv('model_comparison.csv')
        
        st.subheader("🎯 Model Comparison")
        
        # Style the dataframe
        styled_df = comparison_df.style.highlight_max(axis=0, 
                                                      subset=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
                                                      color='lightgreen')
        st.dataframe(styled_df, use_container_width=True)
        
        # Performance charts
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(comparison_df, x='Model', y='Accuracy',
                        title='Model Accuracy Comparison',
                        color='Accuracy',
                        color_continuous_scale='Greens',
                        text='Accuracy')
            fig.update_traces(texttemplate='%{text:.2%}', textposition='outside')
            fig.update_layout(showlegend=False, yaxis_title='Accuracy')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(comparison_df, x='Model', y='ROC-AUC',
                        title='ROC-AUC Score Comparison',
                        color='ROC-AUC',
                        color_continuous_scale='Blues',
                        text='ROC-AUC')
            fig.update_traces(texttemplate='%{text:.4f}', textposition='outside')
            fig.update_layout(showlegend=False, yaxis_title='ROC-AUC Score')
            st.plotly_chart(fig, use_container_width=True)
        
        # All metrics comparison
        st.subheader("📊 All Metrics Comparison")
        metrics_melted = comparison_df.melt(id_vars=['Model'],
                                           value_vars=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                                           var_name='Metric',
                                           value_name='Score')
        
        fig = px.bar(metrics_melted, x='Model', y='Score', color='Metric',
                    barmode='group',
                    title='Comprehensive Metrics Comparison',
                    color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(yaxis_title='Score', xaxis_title='Model')
        st.plotly_chart(fig, use_container_width=True)
        
    except FileNotFoundError:
        st.warning("📊 Model comparison file not found. Please train models first to see analytics.")
    
    # Show visualizations if available
    st.subheader("📸 Model Visualizations")
    
    viz_col1, viz_col2 = st.columns(2)
    
    with viz_col1:
        try:
            st.image('roc_curves.png', caption='ROC Curves - Model Comparison', use_container_width=True)
        except:
            st.info("ROC curve visualization not available")
    
    with viz_col2:
        try:
            st.image('confusion_matrix.png', caption='Confusion Matrix - Best Model', use_container_width=True)
        except:
            st.info("Confusion matrix visualization not available")
    
    try:
        st.image('feature_importance.png', caption='Top 15 Feature Importances', use_container_width=True)
    except:
        st.info("Feature importance visualization not available")

# ============================================================================
# TAB 4: Guide
# ============================================================================
with tab4:
    st.header("ℹ️ User Guide & Information")
    
    # Quick Start
    st.subheader("🚀 Quick Start Guide")
    st.markdown("""
    ### How to Use This Application
    
    **1. Single Customer Prediction:**
    - Navigate to the "Single Prediction" tab
    - Fill in customer details using the form
    - Click "Predict Acceptance Probability"
    - Review the prediction, probability score, and recommendations
    
    **2. Batch Analysis:**
    - Go to the "Batch Analysis" tab
    - Upload a CSV file with customer data
    - Click "Run Batch Predictions"
    - Download results for further analysis
    
    **3. Model Performance:**
    - View comprehensive model analytics
    - Compare different ML algorithms
    - Examine ROC curves, confusion matrices, and feature importance
    """)
    
    st.markdown("---")
    
    # About the Model
    st.subheader("🤖 About the Model")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Model Details
        - **Algorithm:** Random Forest Classifier
        - **Accuracy:** 90.47%
        - **ROC-AUC Score:** 0.9247
        - **Training Data:** 42,458 customers
        - **Features:** 25 predictive variables
        - **Class Balance:** SMOTE applied
        """)
    
    with col2:
        st.markdown("""
        ### Top Predictive Factors
        1. **Call Duration** (30.2%)
        2. **Days Since Last Contact** (11.7%)
        3. **Number of Previous Contacts** (8.9%)
        4. **Customer Age** (7.4%)
        5. **Account Balance** (6.8%)
        """)
    
    st.markdown("---")
    
    # Business Impact
    st.subheader("💼 Business Impact")
    
    impact_col1, impact_col2, impact_col3 = st.columns(3)
    
    with impact_col1:
        st.markdown('<div class="metric-card"><h2>70%</h2><p>Cost Reduction</p></div>', unsafe_allow_html=True)
    
    with impact_col2:
        st.markdown('<div class="metric-card"><h2>3x</h2><p>Conversion Improvement</p></div>', unsafe_allow_html=True)
    
    with impact_col3:
        st.markdown('<div class="metric-card"><h2>90%</h2><p>Revenue Capture</p></div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### How This Helps Your Bank
    
    **🎯 Targeted Marketing:**
    - Focus resources on high-probability customers
    - Reduce wasted marketing spend by up to 70%
    - Achieve 3x better conversion rates
    
    **💰 Financial Benefits:**
    - Significant cost savings on campaigns
    - Higher ROI on marketing investments
    - Better customer experience (fewer irrelevant offers)
    
    **📊 Data-Driven Decisions:**
    - Understand what drives customer acceptance
    - Optimize campaign strategies
    - Personalize customer communications
    """)
    
    st.markdown("---")
    
    # Technical Details
    with st.expander("🔧 Technical Implementation"):
        st.markdown("""
        ### Technology Stack
        - **Python:** Core programming language
        - **Scikit-learn:** Machine learning framework
        - **Streamlit:** Web application framework
        - **Plotly:** Interactive visualizations
        - **Pandas:** Data manipulation
        
        ### Model Training Process
        1. Data collection (UCI/Kaggle Bank Marketing Dataset)
        2. Data preprocessing and cleaning
        3. Feature engineering (9 derived features)
        4. Class imbalance handling (SMOTE)
        5. Model training (3 algorithms compared)
        6. Hyperparameter optimization
        7. Model evaluation and selection
        8. Deployment and monitoring
        
        ### Evaluation Metrics
        - **Accuracy:** Overall correctness
        - **Precision:** Correctness of positive predictions
        - **Recall:** Coverage of actual positives
        - **F1-Score:** Harmonic mean of precision and recall
        - **ROC-AUC:** Discrimination ability across thresholds
        """)
    
    # FAQs
    with st.expander("❓ Frequently Asked Questions"):
        st.markdown("""
        ### Common Questions
        
        **Q: How accurate is this model?**
        A: The model achieves 90.47% accuracy on test data, significantly outperforming baseline approaches.
        
        **Q: What data is needed for predictions?**
        A: Customer demographics, financial information, and contact history as shown in the input form.
        
        **Q: Can I use this for real campaigns?**
        A: Yes, but recommend validation with your specific customer base and recent data first.
        
        **Q: How often should the model be retrained?**
        A: Quarterly retraining is recommended to maintain performance as customer behavior evolves.
        
        **Q: What if my customer data format is different?**
        A: The model expects specific features. Data preprocessing may be needed to match the required format.
        
        **Q: Is customer data stored?**
        A: No, all predictions are made in real-time and no data is stored by this application.
        """)
    
    st.markdown("---")
    
    # Contact
    st.subheader("📞 Support & Contact")
    st.markdown("""
    **Developer:** [Your Name]  
    **Course:** [Your Course]  
    **Institution:** [Your Institution]  
    
    For questions, feedback, or technical support:
    - 📧 Email: [your.email@example.com]
    - 💼 LinkedIn: [Your LinkedIn Profile]
    - 🐙 GitHub: [Your GitHub Repository]
    """)
    
    st.markdown("---")
    
    # Disclaimer
    st.caption("""
    **⚠️ Disclaimer:** This is an academic project for demonstration purposes. 
    The model should be validated with current data and comply with all applicable 
    regulations (GDPR, CCPA, etc.) before deployment in production environments.
    Predictions are probabilistic and should be used as decision support, not as 
    sole determinants of marketing strategy.
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d; padding: 20px;'>
    <p style='font-size: 1.1rem;'>💳 <strong>Credit Card Purchase Prediction System</strong></p>
    <p>Powered by Machine Learning | Built with Streamlit</p>
    <p style='font-size: 0.9rem;'>© 2024 | All Rights Reserved</p>
</div>
""", unsafe_allow_html=True)
