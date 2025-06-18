import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import Predictions
import Recommandations
import Indicateurs
import utils

# Configuration de base
st.set_page_config(
    page_title="Dashboard Détection de Fraude - Vesta", 
    page_icon="🚨", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Chargement des données
@st.cache_data
def load_data():
    try:
        data = pd.read_csv("data/data_dashboard.csv")
        return data
    except FileNotFoundError:
        st.warning("Fichier de données non trouvé. Utilisation de données simulées.")
        return utils.generate_sample_data()

# Sidebar de filtres
def sidebar_filters(df):
    st.sidebar.title("🚨 Filtres Anti-Fraude")
    
    # Filtre par type de fraude
    fraud_filter = st.sidebar.selectbox(
        "Type de Transaction", 
        options=['Toutes', 'Frauduleuses', 'Légitimes'],
        index=0
    )
    
    # Filtre par ProductCD
    if 'ProductCD' in df.columns:
        product_types = df['ProductCD'].dropna().unique()
        product_filter = st.sidebar.multiselect(
            "Type de Produit", 
            options=product_types, 
            default=product_types
        )
    else:
        product_filter = None
    
    # Filtre par card4 (type de carte)
    if 'card4' in df.columns:
        card_types = df['card4'].dropna().unique()
        card_filter = st.sidebar.multiselect(
            "Type de Carte", 
            options=card_types, 
            default=card_types
        )
    else:
        card_filter = None
    
    # Filtre par DeviceType
    if 'DeviceType' in df.columns:
        device_types = df['DeviceType'].dropna().unique()
        device_filter = st.sidebar.multiselect(
            "Type d'Appareil", 
            options=device_types, 
            default=device_types
        )
    else:
        device_filter = None
    
    # Filtre par montant de transaction
    if 'TransactionAmt' in df.columns:
        min_amt, max_amt = float(df['TransactionAmt'].min()), float(df['TransactionAmt'].max())
        amount_range = st.sidebar.slider(
            "Montant de Transaction ($)", 
            min_value=min_amt, 
            max_value=max_amt, 
            value=(min_amt, max_amt)
        )
    else:
        amount_range = None
    
    # Application des filtres
    filtered_df = df.copy()
    
    # Filtre fraude
    if fraud_filter == 'Frauduleuses':
        filtered_df = filtered_df[filtered_df['isFraud'] == 1]
    elif fraud_filter == 'Légitimes':
        filtered_df = filtered_df[filtered_df['isFraud'] == 0]
    
    # Autres filtres
    if product_filter and 'ProductCD' in df.columns:
        filtered_df = filtered_df[filtered_df['ProductCD'].isin(product_filter)]
    
    if card_filter and 'card4' in df.columns:
        filtered_df = filtered_df[filtered_df['card4'].isin(card_filter)]
    
    if device_filter and 'DeviceType' in df.columns:
        filtered_df = filtered_df[filtered_df['DeviceType'].isin(device_filter)]
    
    if amount_range and 'TransactionAmt' in df.columns:
        filtered_df = filtered_df[
            (filtered_df['TransactionAmt'] >= amount_range[0]) &
            (filtered_df['TransactionAmt'] <= amount_range[1])
        ]
    
    # Affichage des métriques de filtrage
    st.sidebar.markdown("---")
    st.sidebar.metric("Transactions sélectionnées", filtered_df.shape[0])
    
    if 'isFraud' in filtered_df.columns:
        fraud_count = filtered_df['isFraud'].sum()
        fraud_rate = (fraud_count / len(filtered_df) * 100) if len(filtered_df) > 0 else 0
        st.sidebar.metric("Fraudes détectées", f"{fraud_count}")
        st.sidebar.metric("Taux de fraude", f"{fraud_rate:.2f}%")
    
    return filtered_df

# Main
def main():
    # Chargement des données
    df = load_data()
    
    
    st.markdown("""
<style>
  
    /* Cible uniquement la barre de navigation principale */
    div[data-testid="stTabs"] > div > div > div {
        display: flex !important;
        justify-content: space-around !important;
        gap: 20 px!;
        background-color: #f5f5f5 !important;  /* Gris très clair */
        border-radius: 12px !important;
        padding: 0 10px !important;
    }
    
    /* Répartition égale des onglets */
    div[data-testid="stTabs"] > div > div > div > button {
        flex: 1 !important;
        max-width: calc(100%/3) !important;
        margin: 0 !important;
    }

    
    /* Texte des onglets */
    div[data-testid="stTabs"] > div > div > div > button > div > p {
        font-size: 20px !important;  /* Taille augmentée */
        font-weight: 600;
        font-family: 'Arial', sans-serif;
    }
    
    /* Onglet actif */
    div[data-testid="stTabs"] > div > div > div > button[aria-selected="true"] > div > p {
        color: white !important;
        font-size: 22px !important;
    }
    
    /* Fond onglet actif */
    div[data-testid="stTabs"] > div > div > div > button[aria-selected="true"] {
        background-color: #d63031 !important;
        border-radius: 8px !important;
    }
    
    h2, h3 {

        background-color: #d63031 !important;  /* Fond rouge */
        color: white !important;              /* Texte blanc */
        padding: 12px 16px !important;
        border-radius: 8px !important;
        font-size: 1.3rem !important;
        font-weight: 600 !important;
        margin: 1.5rem 0 0.8rem 0 !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
        border: none !important;
        text-align: center !important;        /* Centrage du texte */
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
    }
    
    /* Centrage spécifique pour les icônes + texte */
    h2 span, h3 span {
        display: inline-flex !important;
        align-items: center !important;
        gap: 8px !important;  /* Espace entre icône et texte */
    }
    
    /* Espacement après les sections */
    div.stContainer {
        margin-bottom: 1.5rem !important;
    }
    
    /* Style pour le contenu */
    div[data-testid="stMarkdownContainer"] {
        padding: 0.5rem 1rem !important;
    }
    
    
</style>
""", unsafe_allow_html=True)# Ajout d'un style CSS pour personnaliser la barre de navigation
   
    
    # Titre principal avec icône
    st.markdown("""
    <div style='display: flex; justify-content: center; margin-bottom: 3rem;'>
        <div style='background-color: #f5f5f5; padding: 2rem; border-radius: 12px;
                    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1); width: 100%; '>
            <h1 style='color: #d63031; font-size: 2.8rem; text-align: center; margin: 0;'>
                🕵️‍♂️ DASHBOARD ANTI-FRAUDE POUR VESTA CORPORATION
            </h1>
        </div>
    </div>
""", unsafe_allow_html=True)


    
    # Création des onglets de navigation
    tabs = st.tabs(["📊 Indicateurs & KPIs", "🔮 Prédictions", "💡 Recommandations"])
    
    # Application des filtres
    filtered_df = sidebar_filters(df)
    
    # Affichage du contenu selon l'onglet sélectionné
    with tabs[0]:
        Indicateurs.indicateurs_page(filtered_df)
    
    with tabs[1]:
        Predictions.predictions_page(filtered_df)
    
    with tabs[2]:
        Recommandations.recommendations_page(filtered_df)

if __name__ == "__main__":
    main()