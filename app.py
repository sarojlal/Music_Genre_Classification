# app.py
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import streamlit as st
import numpy as np
import pandas as pd
import librosa
import pickle
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Music Genre Classifier",
    page_icon="🎵",
    layout="wide"
)

@st.cache_resource
def load_models():
    """Load all saved models and preprocessing objects"""
    try:
        models = {
            'Random Forest': pickle.load(open('saved_models/random_forest_tuned.pkl', 'rb')),
            'SVM': pickle.load(open('saved_models/svm_tuned.pkl', 'rb')),
            'KNN': pickle.load(open('saved_models/knn_tuned.pkl', 'rb')),
            'Logistic Regression': pickle.load(open('saved_models/logistic_regression_tuned.pkl', 'rb')),
            'XGBoost': pickle.load(open('saved_models/xgboost_tuned.pkl', 'rb')),
            'Neural Network': tf.keras.models.load_model('saved_models/nn_complete_model.h5')
        }
        label_encoder = pickle.load(open('saved_models/label_encoder.pkl', 'rb'))
        scaler = pickle.load(open('saved_models/scaler.pkl', 'rb'))
        
        with open('saved_models/feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
            
        return models, label_encoder, scaler, feature_names
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None, None

def extract_features(audio_path):
    """Extract audio features with librosa 0.10.0+ compatibility"""
    try:
        y, sr = librosa.load(audio_path, duration=30)
        
        # Get tempo using current librosa method
        try:
            tempo = float(librosa.feature.rhythm.tempo(y=y, sr=sr)[0])
        except:
            # Fallback for older versions
            tempo = float(librosa.beat.tempo(y=y, sr=sr)[0])
        
        features = {
            'chroma_stft_mean': np.mean(librosa.feature.chroma_stft(y=y, sr=sr)),
            'chroma_stft_var': np.var(librosa.feature.chroma_stft(y=y, sr=sr)),
            'rms_mean': np.mean(librosa.feature.rms(y=y)),
            'rms_var': np.var(librosa.feature.rms(y=y)),
            'spectral_centroid_mean': np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
            'spectral_centroid_var': np.var(librosa.feature.spectral_centroid(y=y, sr=sr)),
            'spectral_bandwidth_mean': np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)),
            'spectral_bandwidth_var': np.var(librosa.feature.spectral_bandwidth(y=y, sr=sr)),
            'rolloff_mean': np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)),
            'rolloff_var': np.var(librosa.feature.spectral_rolloff(y=y, sr=sr)),
            'zero_crossing_rate_mean': np.mean(librosa.feature.zero_crossing_rate(y)),
            'zero_crossing_rate_var': np.var(librosa.feature.zero_crossing_rate(y)),
            'harmony_mean': np.mean(librosa.effects.harmonic(y)),
            'harmony_var': np.var(librosa.effects.harmonic(y)),
            'perceptr_mean': np.mean(librosa.effects.percussive(y)),
            'perceptr_var': np.var(librosa.effects.percussive(y)),
            'tempo': tempo,
        }
        
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        for i in range(20):
            features[f'mfcc{i+1}_mean'] = np.mean(mfcc[i])
            features[f'mfcc{i+1}_var'] = np.var(mfcc[i])
        
        return pd.DataFrame([features])
    except Exception as e:
        st.error(f"Error extracting features: {str(e)}")
        return None

def main():
    st.title("🎵 Music Genre Classifier")
    st.markdown("Upload a music file to predict its genre using 6 ML models")
    
    models, le, scaler, feature_names = load_models()
    if None in [models, le, scaler, feature_names]:
        return
    
    uploaded_file = st.file_uploader(
        "Choose an audio file (WAV, MP3)", 
        type=['wav', 'mp3']
    )
    
    if uploaded_file:
        temp_path = "temp_audio." + uploaded_file.name.split('.')[-1]
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.audio(uploaded_file)
        
        try:
            features_df = extract_features(temp_path)
            if features_df is None:
                return
                
            # Ensure feature alignment
            features_df = features_df.reindex(columns=feature_names, fill_value=0)
            features_scaled = scaler.transform(features_df)
            
            st.subheader("Prediction Results")
            col1, col2 = st.columns([2, 1])
            
            with col1:
                results = []
                for name, model in models.items():
                    try:
                        if name == 'Neural Network':
                            pred_prob = model.predict(features_scaled, verbose=0)[0]
                        else:
                            pred_prob = model.predict_proba(features_scaled)[0]
                        
                        pred_class = np.argmax(pred_prob)
                        results.append({
                            'Model': name,
                            'Genre': le.classes_[pred_class],
                            'Confidence': f"{np.max(pred_prob)*100:.1f}%",
                            'Probs': pred_prob
                        })
                    except Exception as e:
                        st.error(f"{name} error: {str(e)}")
                
                if results:
                    # Display results
                    st.dataframe(
                        pd.DataFrame(results)[['Model', 'Genre', 'Confidence']],
                        height=300
                    )
                    
                    # Detailed probabilities
                    with st.expander("Detailed Probabilities"):
                        for result in results:
                            st.write(f"**{result['Model']}**")
                            prob_df = pd.DataFrame({
                                'Genre': le.classes_,
                                'Probability': result['Probs']
                            }).sort_values('Probability', ascending=False)
                            st.bar_chart(prob_df.set_index('Genre'))
            
            with col2:
                st.write("**Extracted Features**")
                st.dataframe(features_df.T.style.background_gradient(cmap='Blues'))
                
        except Exception as e:
            st.error(f"Processing error: {str(e)}")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

if __name__ == "__main__":
    main()