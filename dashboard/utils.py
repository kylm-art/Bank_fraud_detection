import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Palette de couleurs pour le thème fraude
FRAUD_COLORS = {
    'primary': '#d63031',      # Rouge principal
    'secondary': '#ff6b6b',    # Rouge clair
    'accent': '#fd79a8',       # Rose
    'warning': '#fdcb6e',      # Orange
    'success': '#00b894',      # Vert
    'background': '#2d3436',   # Gris foncé
    'text': '#636e72',         # Gris texte
    'light': '#f8f9fa'         # Blanc cassé
}

def generate_sample_data(n_samples=10000):
    """Génère des données simulées pour la démonstration"""
    np.random.seed(42)
    
    # Génération des données de base
    data = {
        'TransactionID': range(1, n_samples + 1),
        'TransactionAmt': np.random.lognormal(mean=3, sigma=1, size=n_samples),
        'ProductCD': np.random.choice(['C', 'H', 'R', 'S', 'W'], size=n_samples, p=[0.3, 0.2, 0.2, 0.2, 0.1]),
        'card4': np.random.choice(['visa', 'mastercard', 'discover', 'american express'], 
                                 size=n_samples, p=[0.5, 0.3, 0.15, 0.05]),
        'DeviceType': np.random.choice(['desktop', 'mobile'], size=n_samples, p=[0.6, 0.4]),
        'DeviceInfo': np.random.choice(['Windows', 'MacOS', 'iOS', 'Android'], 
                                      size=n_samples, p=[0.4, 0.2, 0.2, 0.2]),
        'addr2': np.random.choice(range(1, 100), size=n_samples)
    }
    
    # Génération des variables id_XX
    id_cols = [f'id_{i:02d}' for i in range(1, 39) if i not in [7, 8, 18, 21, 22, 23, 24, 25, 26, 27]]
    for col in id_cols:
        # Simulation de valeurs manquantes
        values = np.random.normal(0, 1, n_samples)
        missing_mask = np.random.random(n_samples) < 0.1  # 10% de valeurs manquantes
        values[missing_mask] = np.nan
        data[col] = values
    
    # Génération des variables M1-M9
    m_cols = [f'M{i}' for i in range(1, 10)]
    for col in m_cols:
        # Variables binaires avec des valeurs manquantes
        values = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
        missing_mask = np.random.random(n_samples) < 0.05  # 5% de valeurs manquantes
        values = values.astype(float)
        values[missing_mask] = np.nan
        data[col] = values
    
    # Génération de la variable cible (fraude)
    # Probabilité de fraude basée sur certaines caractéristiques
    fraud_prob = 0.02  # 2% de base
    high_amount_boost = (data['TransactionAmt'] > np.percentile(data['TransactionAmt'], 95)) * 0.08
    
    fraud_probs = fraud_prob + high_amount_boost
    data['isFraud'] = np.random.binomial(1, fraud_probs)
    
    return pd.DataFrame(data)

def calculate_fraud_metrics(df):
    """Calcule les métriques clés de fraude"""
    if 'isFraud' not in df.columns:
        return {}
    
    total_transactions = len(df)
    fraud_count = df['isFraud'].sum()
    fraud_rate = (fraud_count / total_transactions * 100) if total_transactions > 0 else 0
    tot_trans_amount = df['TransactionAmt'].sum() if 'TransactionAmt' in df.columns else 0
    
    if 'TransactionAmt' in df.columns:
        fraud_transactions = df[df['isFraud'] == 1]
        min_fraud_amount = fraud_transactions['TransactionAmt'].min() if len(fraud_transactions) > 0 else 0
        avg_fraud_amount = fraud_transactions['TransactionAmt'].mean() if len(fraud_transactions) > 0 else 0
        total_fraud_amount = fraud_transactions['TransactionAmt'].sum() if len(fraud_transactions) > 0 else 0
        transac_amount_rate = (total_fraud_amount /tot_trans_amount) if total_transactions > 0 else 0
    else:
        min_fraud_amount = avg_fraud_amount = total_fraud_amount = 0
    
    return {
        'total_transactions': total_transactions,
        'fraud_count': fraud_count,
        'fraud_rate': fraud_rate,
        'min_fraud_amount': min_fraud_amount,
        'avg_fraud_amount': avg_fraud_amount,
        'total_fraud_amount': total_fraud_amount,
        'transac_amount_rate': transac_amount_rate
    }

def count_missing_variables(df, prefix):
    """Compte le nombre de variables manquantes pour un préfixe donné"""
    cols = [col for col in df.columns if col.startswith(prefix)]
    if not cols:
        return pd.Series(0, index=df.index)
    
    return df[cols].isnull().sum(axis=1)

def create_fraud_distribution_chart(df):
    """Crée un graphique de distribution des montants pour fraudes vs non-fraudes"""
    if 'TransactionAmt' not in df.columns or 'isFraud' not in df.columns:
        return go.Figure()
    
    fraud_data = df[df['isFraud'] == 1]['TransactionAmt']
    legit_data = df[df['isFraud'] == 0]['TransactionAmt']
    
    fig = go.Figure()
    
    # Distribution des transactions frauduleuses
    fig.add_trace(go.Histogram(
        x=fraud_data,
        name='Transactions Frauduleuses',
        opacity=0.7,
        marker_color=FRAUD_COLORS['primary'],
        histnorm='probability density'
    ))
    
    # Distribution des transactions légitimes
    fig.add_trace(go.Histogram(
        x=legit_data,
        name='Transactions Légitimes',
        opacity=0.7,
        marker_color=FRAUD_COLORS['success'],
        histnorm='probability density'
    ))
    
    fig.update_layout(
        title="Distribution des Montants de Transaction (Fraude vs Légitime)",
        xaxis_title="Montant de Transaction ($)",
        yaxis_title="Densité",
        barmode='overlay',
        template='plotly_white',
        showlegend=True
    )
    
    return fig

def create_fraud_rate_map(df):
    """Crée une carte du taux de fraude par zone (addr2)"""
    if 'addr2' not in df.columns or 'isFraud' not in df.columns:
        return go.Figure()
    
    # Calcul du taux de fraude par zone
    zone_stats = df.groupby('addr2').agg({
        'isFraud': ['count', 'sum']
    }).round(3)
    
    zone_stats.columns = ['total_transactions', 'fraud_count']
    zone_stats['fraud_rate'] = (zone_stats['fraud_count'] / zone_stats['total_transactions'] * 100).round(2)
    zone_stats = zone_stats.reset_index()
    
    # Création du graphique en barres (simulant une carte)
    fig = px.bar(
        zone_stats.head(20),  # Top 20 zones
        x='addr2',
        y='fraud_rate',
        color='fraud_rate',
        color_continuous_scale=['green', 'yellow', 'red'],
        title="Taux de Fraude par Zone (Top 20)",
        labels={'addr2': 'Zone', 'fraud_rate': 'Taux de Fraude (%)'}
    )
    
    fig.update_layout(template='plotly_white')
    
    return fig

def create_device_fraud_chart(df):
    """Crée un graphique des fraudes par type d'appareil"""
    if 'DeviceType' not in df.columns or 'isFraud' not in df.columns:
        return go.Figure()
    
    device_fraud = df.groupby('DeviceType')['isFraud'].agg(['count', 'sum']).reset_index()
    device_fraud.columns = ['DeviceType', 'total', 'fraud_count']
    device_fraud['fraud_rate'] = (device_fraud['fraud_count'] / device_fraud['total'] * 100).round(2)
    
    fig = px.pie(
        device_fraud,
        values='fraud_count',
        names='DeviceType',
        title="Répartition des Fraudes par Type d'Appareil",
        color_discrete_sequence=[FRAUD_COLORS['primary'], FRAUD_COLORS['secondary'], FRAUD_COLORS['accent']]
    )
    
    return fig

def create_metric_card(title, value, delta=None, delta_color="normal"):
    """Crée une carte de métrique stylisée"""
    delta_html = ""
    if delta is not None:
        color = "green" if delta_color == "normal" else "red"
        delta_html = f"<p style='color: {color}; font-size: 0.9rem; margin: 0;'>{delta}</p>"
    
    return f"""
    <div class="metric-card">
        <h3 style="margin: 0; color: {FRAUD_COLORS['primary']};">{title}</h3>
        <p style="font-size: 2rem; font-weight: bold; margin: 0.5rem 0; color: {FRAUD_COLORS['background']};">{value}</p>
        {delta_html}
    </div>
    """

def create_product_fraud_analysis(df):
    """Analyse des fraudes par type de produit"""
    if 'ProductCD' not in df.columns or 'isFraud' not in df.columns:
        return go.Figure()
    
    product_stats = df.groupby('ProductCD').agg({
        'isFraud': ['count', 'sum'],
        'TransactionAmt': 'mean'
    }).round(2)
    
    product_stats.columns = ['total_transactions', 'fraud_count', 'avg_amount']
    product_stats['fraud_rate'] = (product_stats['fraud_count'] / product_stats['total_transactions'] * 100).round(2)
    product_stats = product_stats.reset_index()
    
    # Graphique en barres groupées
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Nombre de Fraudes par Produit', 'Taux de Fraude par Produit'),
        vertical_spacing=0.1
    )
    
    # Nombre de fraudes
    fig.add_trace(
        go.Bar(x=product_stats['ProductCD'], y=product_stats['fraud_count'],
               name='Nombre de Fraudes', marker_color=FRAUD_COLORS['primary']),
        row=1, col=1
    )
    
    # Taux de fraude
    fig.add_trace(
        go.Bar(x=product_stats['ProductCD'], y=product_stats['fraud_rate'],
               name='Taux de Fraude (%)', marker_color=FRAUD_COLORS['secondary']),
        row=2, col=1
    )
    
    fig.update_layout(
        height=600,
        title="Analyse des Fraudes par Type de Produit",
        template='plotly_white',
        showlegend=False
    )
    
    return fig