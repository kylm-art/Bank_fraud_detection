import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import utils

def predictions_page(df):
    """Page de prédictions de fraude"""
    
    st.markdown("## 🔮 Prédictions Anti-Fraude")
    st.markdown("*Interface de prédiction en temps réel et analyse de fichiers*")
    
    # === SECTION 1: FEATURES IMPORTANTES ===
    st.markdown("### 📊 Variables les Plus Importantes")
    
    # Simulation des features importantes (en attendant le modèle réel)
    feature_importance = generate_feature_importance(df)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig_importance = px.bar(
            feature_importance,
            x='importance',
            y='feature',
            orientation='h',
            title="Importance des Variables dans la Prédiction",
            color='importance',
            color_continuous_scale=['lightblue', 'darkblue']
        )
        fig_importance.update_layout(
            template='plotly_white',
            height=400,
            yaxis={'title': 'Variables'},
            xaxis={'title': 'Score d\'Importance'}
        )
        st.plotly_chart(fig_importance, use_container_width=True)
    
    with col2:
        st.markdown("#### 🎯 Top 5 Prédicteurs")
        for i, row in feature_importance.head(5).iterrows():
            st.markdown(f"- **{row['feature']}**: {row['importance']:.2f}")
    st.markdown("---")
    # === SECTION 2: PRÉDICTION EN TEMPS RÉEL ===
    st.markdown("### 🕒 Prédiction en Temps Réel")
    st.markdown("**Entrez les détails de la transaction pour prédire la probabilité de fraude.**")
    with st.form("prediction_form"):
        transaction_data = {
            'TransactionAmt': st.number_input("Montant de la Transaction ($)", min_value=0.0, step=0.01),
            'ProductCD': st.selectbox("Type de Produit", options=df['ProductCD'].unique(), index=0),
            'card4': st.selectbox("Type de Carte", options=df['card4'].unique(), index=0),
            'DeviceType': st.selectbox("Type d'Appareil", options=df['DeviceType'].unique(), index=0),
            # Ajoutez d'autres champs nécessaires
        }
        
        submit_button = st.form_submit_button("Prédire")
        
        if submit_button:
            prediction = predict_transaction(transaction_data)
            st.success(f"Probabilité de Fraude: {prediction['fraud_probability']:.2f}") 
            st.markdown(f"**Estimation de Fraude:** {'Oui' if prediction['is_fraud'] else 'Non'}")
            st.markdown("**Top 5 Variables Impactantes:**")
            for feature in prediction['top_features']:
                st.markdown(f"- **{feature['feature']}**: {feature['importance']:.2f}")
    st.markdown("---")
    # === SECTION 3: ANALYSE DE FICHIERS ===
    st.markdown("### 📂 Analyse de Fichiers")
    st.markdown("**Téléchargez un fichier CSV ou Parquet pour prédire les fraudes en masse.**")
    uploaded_file = st.file_uploader("Choisissez un fichier", type=['csv', 'parquet'])
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.parquet'):
            data = pd.read_parquet(uploaded_file)
        else:
            st.error("Format de fichier non supporté. Veuillez télécharger un fichier CSV ou Parquet.")
            return
        
        # Prétraitement des données
        data = preprocess_data(data)
        
        # Prédictions en masse
        predictions = batch_predict(data)
        
        st.markdown("#### Résultats des Prédictions")
        st.dataframe(predictions[['TransactionID', 'fraud_probability', 'is_fraud']])
        st.markdown("**Top 5 Variables Impactantes pour chaque transaction :**")
        for i, row in predictions.iterrows():
            st.markdown(f"**Transaction ID {row['TransactionID']}:**")
            for feature in row['top_features']:
                st.markdown(f"- **{feature['feature']}**: {feature['importance']:.2f}")
    st.markdown("---")  
def generate_feature_importance(df):
    """Génère une importance de features simulée pour la démonstration"""
    features = df.columns[df.columns.str.startswith('id_') | df.columns.str.startswith('m_')]
    importance = np.random.rand(len(features))
    return pd.DataFrame({'feature': features, 'importance': importance}).sort_values(by='importance', ascending=False)
def predict_transaction(transaction_data):
    """Simule une prédiction de fraude pour une transaction donnée"""
    # Simuler une probabilité de fraude
    fraud_probability = np.random.rand()
    is_fraud = fraud_probability > 0.5  # Seuil arbitraire pour la détection de fraude
    
    # Simuler les features importantes
    top_features = [
        {'feature': 'TransactionAmt', 'importance': np.random.rand()},
        {'feature': 'ProductCD', 'importance': np.random.rand()},
        {'feature': 'card4', 'importance': np.random.rand()},
        {'feature': 'DeviceType', 'importance': np.random.rand()},
        {'feature': 'addr2', 'importance': np.random.rand()}
    ]
    
    return {
        'fraud_probability': fraud_probability,
        'is_fraud': is_fraud,
        'top_features': top_features
    }
