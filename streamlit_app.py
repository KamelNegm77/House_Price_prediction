import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import os

FEATURE_NAMES = [
    'living area', 'number of bedrooms', 'number of bathrooms',
    'grade of the house', 'condition of the house', 'Number of schools nearby',
    'Distance from the airport', 'price_per_sqft'
]

@st.cache_resource
def load_model_and_features_auto():
    # Search in the same directory as the Streamlit file
    model_paths = sorted(Path('.').glob('best_ridge_*.joblib'))
    
    if not model_paths:
        st.warning('⚠️ No Ridge model found. Please ensure a file named best_ridge_*.joblib exists.')
        return None, None, None

    model_path = model_paths[-1]  # load latest
    model = joblib.load(model_path)

    # Prefer explicit features file if present; else use known feature list
    features_path = Path('selected_features.joblib')
    if features_path.exists():
        selected_features = joblib.load(features_path)
        st.success('✅ Loaded selected_features.joblib')
    else:
        st.warning('⚠️ selected_features.joblib not found, using default FEATURE_NAMES.')
        selected_features = FEATURE_NAMES

    return model, selected_features, str(model_path)


def main():
    st.title('🏠 House Price Prediction (Ridge)')
    st.write('Enter property details to predict price using your tuned Ridge model.')

    load_mode = st.sidebar.radio('Model source', ['Auto-detect latest', 'Upload .joblib'], index=0)

    model = None
    selected_features = FEATURE_NAMES
    model_label = ''

    if load_mode == 'Auto-detect latest':
        model, selected_features, model_path = load_model_and_features_auto()
        if model is not None:
            model_label = f'✅ Loaded model: {model_path}'
        else:
            model_label = '❌ No best_ridge_*.joblib found. Please upload a model on the left.'
    else:
        uploaded = st.sidebar.file_uploader('Upload Ridge model (.joblib)', type=['joblib'])
        if uploaded is not None:
            try:
                model = joblib.load(uploaded)
                model_label = '✅ Loaded uploaded model file'
            except Exception as e:
                st.sidebar.error(f'Failed to load model: {e}')

        uploaded_feats = st.sidebar.file_uploader('Optional: selected_features.joblib', type=['joblib'])
        if uploaded_feats is not None:
            try:
                selected_features = joblib.load(uploaded_feats)
            except Exception as e:
                st.sidebar.error(f'Failed to load feature list: {e}')

    st.caption(model_label)

    # Build UI inputs for each selected feature. These are numeric in the notebook.
    defaults = {
        'living area': 1800.0,
        'number of bedrooms': 3.0,
        'number of bathrooms': 2.0,
        'grade of the house': 7.0,
        'condition of the house': 3.0,
        'Number of schools nearby': 3.0,
        'Distance from the airport': 20.0,
        'price_per_sqft': 300.0,
        'house_age': 20.0,
        'renovation_age': 0.0,
        'total_area': 2000.0,
    }

    cols = st.columns(2)
    values = {}
    for i, feat in enumerate(selected_features):
        col = cols[i % 2]
        with col:
            values[feat] = st.number_input(
                feat,
                value=float(defaults.get(feat, 0.0)),
                step=1.0,
                format='%f'
            )

    if st.button('Predict Price'):
        if model is None:
            st.error('❌ No model loaded. Please use Auto-detect (after saving) or upload a .joblib model.')
            return
        try:
            df = pd.DataFrame([{k: values[k] for k in selected_features}])
            pred = model.predict(df)[0]
            st.success(f'💰 Estimated Price: ${pred:,.0f}')
        except Exception as e:
            st.error(f'Prediction failed: {e}')

    with st.expander('Feature help'):
        st.markdown(
            '- **living area**: Interior living space (sqft)\n'
            '- **total_area**: living area + basement area (sqft)\n'
            '- **house_age**: years since built\n'
            '- **renovation_age**: years since last renovation (0 if never)\n'
            '- **price_per_sqft**: estimated market $ per sqft for similar homes\n'
        )

    # Optional: Debug info
    with st.expander("🛠 Debug info"):
        st.write("📂 Current directory:", os.getcwd())
        st.write("📄 Files in directory:", os.listdir('.'))


if __name__ == '__main__':
    main()
