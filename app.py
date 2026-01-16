"""
Streamlit Web Interface for AutoML System
Save as: app.py
Run with: streamlit run app.py

This imports and uses the AutoMLEngine from automl.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from automl import AutoMLEngine  # Import the core engine
import os

# Page configuration
st.set_page_config(
    page_title="AutoML System",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        font-size: 1.1rem;
    }
    .success-box {
        background: #e8f5e9;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #4caf50;
        margin: 1rem 0;
    }
    .info-box {
        background: #e3f2fd;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #2196f3;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'engine' not in st.session_state:
    st.session_state.engine = None
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'data_prepared' not in st.session_state:
    st.session_state.data_prepared = False
if 'training_complete' not in st.session_state:
    st.session_state.training_complete = False


def reset_session():
    """Reset all session state variables."""
    st.session_state.engine = None
    st.session_state.data_loaded = False
    st.session_state.data_prepared = False
    st.session_state.training_complete = False
    if 'selected_features' in st.session_state:
        del st.session_state['selected_features']
    if 'manual_mode' in st.session_state:
        del st.session_state['manual_mode']


def main():
    # Check if automl.py exists
    if not os.path.exists('automl.py'):
        st.error("""
        ❌ **`automl.py` not found!**
        
        Please make sure both files are in the same directory:
        - `automl.py` (the core ML engine)
        - `app.py` (this web interface)
        """)
        st.stop()
    
    st.title("🤖 AutoML - Supervised Learning System")
    st.markdown("### Fully automated machine learning - no coding required!")
    
    # Sidebar
    with st.sidebar:
        st.header("📋 Navigation")
        page = st.radio("", ["📁 Upload Data", "🚀 Train Models", "🎯 Predictions"], 
                       label_visibility="collapsed")
        
        st.markdown("---")
        st.subheader("📊 Progress")
        st.write("✅ Data Loaded" if st.session_state.data_loaded else "⬜ Data Loaded")
        st.write("✅ Data Prepared" if st.session_state.data_prepared else "⬜ Data Prepared")
        st.write("✅ Models Trained" if st.session_state.training_complete else "⬜ Models Trained")
        
        # Reset button
        st.markdown("---")
        if st.button("🔄 Start New Project", use_container_width=True):
            reset_session()
            st.rerun()
        
        if st.session_state.engine and st.session_state.training_complete:
            st.markdown("---")
            if st.button("💾 Save Model", use_container_width=True):
                try:
                    st.session_state.engine.save_model('best_model.pkl')
                    with open('best_model.pkl', 'rb') as f:
                        st.download_button(
                            "📥 Download Model",
                            data=f,
                            file_name="best_model.pkl",
                            mime="application/octet-stream"
                        )
                except Exception as e:
                    st.error(f"Error saving model: {e}")
    
    # ==================== PAGE 1: UPLOAD DATA ====================
    if page == "📁 Upload Data":
        st.header("📁 Step 1: Upload and Configure Dataset")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'], key='file_uploader')
        
        if uploaded_file:
            # Check if this is a new file - reset if so
            if 'last_uploaded_file' not in st.session_state or st.session_state.last_uploaded_file != uploaded_file.name:
                reset_session()
                st.session_state.last_uploaded_file = uploaded_file.name
            
            # Load data into engine
            df = pd.read_csv(uploaded_file)
            st.session_state.engine = AutoMLEngine(dataframe=df)
            st.session_state.data_loaded = True
            
            st.success(f"✅ Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
            
            # Preview
            with st.expander("📊 View Data Preview", expanded=True):
                st.dataframe(df.head(10), use_container_width=True)
            
            # Statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Rows", df.shape[0])
            with col2:
                st.metric("Columns", df.shape[1])
            with col3:
                st.metric("Missing", df.isnull().sum().sum())
            with col4:
                st.metric("Numeric", df.select_dtypes(include=[np.number]).shape[1])
            
            st.markdown("---")
            
            # Target column selection
            st.subheader("🎯 Select Target Column")
            
            detected_target, candidates = st.session_state.engine.detect_target_column()
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.info(f"🔍 **Auto-detected:** `{detected_target}`")
            with col2:
                with st.expander("Top 3 Candidates"):
                    for col, score in candidates:
                        st.write(f"• {col} (score: {score})")
            
            target_col = st.selectbox(
                "Confirm or select different target column:",
                options=df.columns.tolist(),
                index=df.columns.tolist().index(detected_target)
            )
            
            # Feature selection
            st.markdown("---")
            st.subheader("🔧 Select Features for Training")
            
            available_features = [col for col in df.columns if col != target_col]
            
            # Show feature analysis
            feature_info = []
            for col in available_features:
                unique_ratio = df[col].nunique() / len(df)
                is_id = unique_ratio > 0.5 and df[col].dtype == 'object'
                feature_info.append({
                    'Column': col,
                    'Type': str(df[col].dtype),
                    'Unique': df[col].nunique(),
                    'Missing': df[col].isnull().sum(),
                    'Recommendation': '❌ Remove (ID)' if is_id else '✅ Keep'
                })
            
            st.dataframe(pd.DataFrame(feature_info), use_container_width=True)
            
            # Selection buttons
            col1, col2, col3 = st.columns(3)
            
            selected_features = None
            
            with col1:
                if st.button("🤖 Auto Select", use_container_width=True):
                    selected_features, removed = st.session_state.engine.auto_select_features(target_col)
                    if removed:
                        st.warning(f"🗑️ Removed: {', '.join(removed)}")
                    st.success(f"✅ Selected {len(selected_features)} features")
                    st.session_state['selected_features'] = selected_features
            
            with col2:
                if st.button("✅ Select All", use_container_width=True):
                    selected_features = available_features
                    st.success(f"✅ Selected all {len(selected_features)} features")
                    st.session_state['selected_features'] = selected_features
            
            with col3:
                if st.button("✋ Manual", use_container_width=True):
                    st.session_state['manual_mode'] = True
            
            # Manual selection
            if st.session_state.get('manual_mode'):
                selected_features = st.multiselect(
                    "Choose features:",
                    available_features,
                    default=available_features
                )
                st.session_state['selected_features'] = selected_features
            
            # Show selected features
            if 'selected_features' in st.session_state and st.session_state['selected_features']:
                selected_features = st.session_state['selected_features']
                st.info(f"**Selected features ({len(selected_features)}):** {', '.join(selected_features)}")
                
                st.markdown("---")
                
                if st.button("✅ Prepare Data for Training", type="primary", use_container_width=True):
                    if len(selected_features) == 0:
                        st.error("❌ Please select at least one feature!")
                    else:
                        with st.spinner("🔄 Preparing data..."):
                            st.session_state.engine.prepare_data(target_col, selected_features)
                            st.session_state.data_prepared = True
                        
                        st.success("✅ Data prepared successfully!")
                        st.info(f"**Problem Type:** {st.session_state.engine.problem_type}")
                        st.balloons()
    
    # ==================== PAGE 2: TRAIN MODELS ====================
    elif page == "🚀 Train Models":
        st.header("🚀 Step 2: Train All Models")
        
        if not st.session_state.data_prepared:
            st.warning("⚠️ Please upload and prepare data first!")
            st.info("👈 Go to 'Upload Data' to get started")
            return
        
        engine = st.session_state.engine
        
        # Show data info
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Train Samples", len(engine.X_train))
        with col2:
            st.metric("Test Samples", len(engine.X_test))
        with col3:
            st.metric("Features", len(engine.feature_columns))
        with col4:
            st.metric("Type", engine.problem_type)
        
        st.markdown("---")
        
        if not st.session_state.training_complete:
            st.markdown("""
            <div class="info-box">
            <h4>🎯 Ready to train!</h4>
            <p>The system will train 11 different ML algorithms and compare their performance.</p>
            <p>⏱️ This usually takes 30 seconds to 2 minutes.</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("🚀 Start Training", type="primary", use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def progress_callback(current, total, model_name):
                    progress_bar.progress(current / total)
                    status_text.text(f"Training {model_name}... ({current}/{total})")
                
                # Train models
                results = engine.train_all_models(progress_callback=progress_callback)
                
                st.session_state.training_complete = True
                
                status_text.text("✅ Training completed!")
                st.success("✅ All models trained successfully!")
                st.balloons()
                st.rerun()
        
        else:
            # Show results
            st.success("✅ Training Complete!")
            
            # Check if results exist
            if not engine.results:
                st.error("❌ No results available. Please retrain the models.")
                if st.button("🔄 Reset and Start Over"):
                    reset_session()
                    st.rerun()
                return
            
            results_df = engine.get_results_dataframe()
            
            # Check if results_df is valid
            if results_df is None or results_df.empty:
                st.error("❌ Results are empty. Please retrain the models.")
                if st.button("🔄 Reset and Start Over"):
                    reset_session()
                    st.rerun()
                return
            
            # Results table
            st.subheader("📊 Model Performance Comparison")
            
            # Format numeric columns
            display_df = results_df.copy()
            for col in display_df.columns:
                if col != 'Model' and display_df[col].dtype in ['float64', 'int64']:
                    display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")
            
            st.dataframe(display_df, use_container_width=True)
            
            # Best model details
            st.markdown("---")
            st.subheader(f"🏆 Best Model: {engine.best_model_name}")
            
            best_result = engine.results[0]
            
            if 'Classification' in engine.problem_type:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Accuracy", f"{best_result['Accuracy']*100:.2f}%")
                with col2:
                    st.metric("Precision", f"{best_result['Precision']*100:.2f}%")
                with col3:
                    st.metric("Recall", f"{best_result['Recall']*100:.2f}%")
                with col4:
                    st.metric("F1 Score", f"{best_result['F1_Score']*100:.2f}%")
            else:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("R² Score", f"{best_result['R2_Score']:.4f}")
                with col2:
                    st.metric("RMSE", f"{best_result['RMSE']:.4f}")
                with col3:
                    st.metric("MAE", f"{best_result['MAE']:.4f}")
    
    # ==================== PAGE 3: PREDICTIONS ====================
    elif page == "🎯 Predictions":
        st.header("🎯 Step 3: Make Predictions")
        
        if not st.session_state.training_complete:
            st.warning("⚠️ Please train models first!")
            st.info("👈 Go to 'Train Models' to train your models")
            return
        
        engine = st.session_state.engine
        
        st.success(f"Using: **{engine.best_model_name}**")
        
        st.subheader("Enter Feature Values")
        
        input_data = {}
        
        # Create input fields
        cols = st.columns(2)
        for i, feature in enumerate(engine.feature_columns):
            with cols[i % 2]:
                # Check if categorical
                if feature in engine.label_encoders:
                    classes = engine.label_encoders[feature].classes_
                    input_data[feature] = st.selectbox(f"**{feature}**", classes, key=feature)
                else:
                    # Numeric input
                    input_data[feature] = st.number_input(f"**{feature}**", value=0.0, key=feature)
        
        st.markdown("---")
        
        if st.button("🎯 Get Prediction", type="primary", use_container_width=True):
            with st.spinner("Making prediction..."):
                prediction = engine.predict(input_data)
            
            st.markdown("### 🎉 Prediction Result")
            st.markdown(f"""
            <div class="success-box">
            <h1 style="text-align: center; color: #4caf50; margin: 0;">
            {prediction}
            </h1>
            </div>
            """, unsafe_allow_html=True)
            
            st.balloons()


if __name__ == "__main__":
    main()












































































































