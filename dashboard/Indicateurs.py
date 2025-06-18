import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import utils

def indicateurs_page(df):
    """Page principale des indicateurs de fraude"""
   
    # Calcul des métriques principales
    metrics = utils.calculate_fraud_metrics(df)
    
    # === SECTION 1: CHIFFRES CLÉS ===
    st.markdown("## 🔢 Chiffres Clés")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(utils.create_metric_card(
            "Total Transactions", 
            f"{metrics['total_transactions']:,}"
        ), unsafe_allow_html=True)
    
    with col2:
        st.markdown(utils.create_metric_card(
            "Fraudes Détectées", 
            f"{metrics['fraud_count']:,}",
            f"🚨 {metrics['fraud_rate']:.2f}% du total"
        ), unsafe_allow_html=True)
    
    with col3:
        if metrics['avg_fraud_amount'] > 0:
            st.markdown(utils.create_metric_card(
                "Montant Moyen Fraude", 
                f"${metrics['avg_fraud_amount']:,.2f}"
            ), unsafe_allow_html=True)
        else:
            st.markdown(utils.create_metric_card("Montant Moyen Fraude", "N/A"), unsafe_allow_html=True)
    
    with col4:
        if metrics['transac_amount_rate'] > 0:
            st.markdown(utils.create_metric_card(
                "Montant Total Fraude", 
                f"${metrics['total_fraud_amount']:,.2f}",
                f"({metrics['transac_amount_rate']:.2f}% du total)"
            ), unsafe_allow_html=True)
        else:
            st.markdown(utils.create_metric_card("Montant Min. Fraude", "N/A"), unsafe_allow_html=True)
    
    st.markdown("---")
    
    # === SECTION 2: ANALYSES PAR VARIABLES ID ===
    st.markdown("### 🆔 Données d'identification et fraude")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Analyse des variables id manquantes
        if any(col.startswith('id_') for col in df.columns):
            df['missing_id_count'] = utils.count_missing_variables(df, 'id_')
            
            # Fraudes selon le nombre d'ID manquants
            id_fraud_analysis = df.groupby('missing_id_count')['isFraud'].agg(['count', 'sum']).reset_index()
            id_fraud_analysis.columns = ['missing_ids', 'total', 'fraud_count']
            id_fraud_analysis['fraud_rate'] = (id_fraud_analysis['fraud_count'] / id_fraud_analysis['total'] * 100).round(2)
            
            fig_id = px.bar(
                id_fraud_analysis,
                x='missing_ids',
                y='fraud_count',
                color='fraud_rate',
                color_continuous_scale=['lightgreen', 'yellow', 'red'],
                title="Fraudes selon le Nombre d'IDs Manquants",
                labels={'missing_ids': 'Nombre d\'IDs Manquants', 'fraud_count': 'Nombre de Fraudes'}
            )
            fig_id.update_layout(template='plotly_white')
            st.plotly_chart(fig_id, use_container_width=True)
        else:
            st.info("Aucune variable ID disponible dans les données")
    
    with col2:
        # Distribution des IDs manquants
        if 'missing_id_count' in df.columns:
            fig_id_dist = px.histogram(
                df,
                x='missing_id_count',
                color='isFraud',
                color_discrete_map={0: utils.FRAUD_COLORS['success'], 1: utils.FRAUD_COLORS['primary']},
                title="Distribution des IDs Manquants",
                labels={'missing_id_count': 'Nombre d\'IDs Manquants', 'count': 'Nombre de Transactions'}
            )
            fig_id_dist.update_layout(template='plotly_white')
            st.plotly_chart(fig_id_dist, use_container_width=True)
    
    # === SECTION 3: ANALYSES DES VARIABLES M ===
    st.markdown("### correspondance des cartes et fraudes")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Variables M manquantes
        if any(col.startswith('M') and col[1:].isdigit() for col in df.columns):
            df['missing_m_count'] = utils.count_missing_variables(df, 'M')
            
            m_fraud_analysis = df.groupby('missing_m_count')['isFraud'].agg(['count', 'sum']).reset_index()
            m_fraud_analysis.columns = ['missing_ms', 'total', 'fraud_count']
            m_fraud_analysis['fraud_rate'] = (m_fraud_analysis['fraud_count'] / m_fraud_analysis['total'] * 100).round(2)
            
            fig_m = px.line(
                m_fraud_analysis,
                x='missing_ms',
                y='fraud_rate',
                markers=True,
                title="Taux de Fraude selon les Variables M Manquantes",
                labels={'missing_ms': 'Nombre de Variables M Manquantes', 'fraud_rate': 'Taux de Fraude (%)'}
            )
            fig_m.update_traces(line_color=utils.FRAUD_COLORS['primary'], marker_color=utils.FRAUD_COLORS['primary'])
            fig_m.update_layout(template='plotly_white')
            st.plotly_chart(fig_m, use_container_width=True)
    
    with col2:
        # Variables M avec valeur False
        if any(col.startswith('M') and col[1:].isdigit() for col in df.columns):
            m_cols = [col for col in df.columns if col.startswith('M') and col[1:].isdigit()]
            df['false_m_count'] = df[m_cols].eq(0).sum(axis=1)
            
            m_false_analysis = df.groupby('false_m_count')['isFraud'].agg(['count', 'sum']).reset_index()
            m_false_analysis.columns = ['false_ms', 'total', 'fraud_count']
            m_false_analysis['fraud_rate'] = (m_false_analysis['fraud_count'] / m_false_analysis['total'] * 100).round(2)
            
            fig_m_false = px.scatter(
                m_false_analysis,
                x='false_ms',
                y='fraud_rate',
                size='total',
                color='fraud_rate',
                color_continuous_scale=['green', 'yellow', 'red'],
                title="Taux de Fraude selon les Variables M = False",
                labels={'false_ms': 'Nombre de Variables M = False', 'fraud_rate': 'Taux de Fraude (%)'}
            )
            fig_m_false.update_layout(template='plotly_white')
            st.plotly_chart(fig_m_false, use_container_width=True)
    
    st.markdown("---")
    
    # === SECTION 4: ANALYSES DES DISPOSITIFS ===
    st.markdown("### Dispositif de connexion et fraude")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Analyse par DeviceInfo (OS)
        if 'DeviceInfo' in df.columns:
            device_info_stats = df.groupby('DeviceInfo').agg({
                'isFraud': ['count', 'sum']
            }).round(2)
            device_info_stats.columns = ['total', 'fraud_count']
            device_info_stats['fraud_rate'] = (device_info_stats['fraud_count'] / device_info_stats['total'] * 100).round(2)
            device_info_stats = device_info_stats.reset_index()
            
            fig_os = px.pie(
                device_info_stats,
                values='fraud_count',
                names='DeviceInfo',
                title="Répartition des Fraudes par Système d'Exploitation",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig_os, use_container_width=True)
        else:
            st.info("Données DeviceInfo non disponibles")
    
    with col2:
        # Analyse par DeviceType
        if 'DeviceType' in df.columns:
            fig_device = utils.create_device_fraud_chart(df)
            st.plotly_chart(fig_device, use_container_width=True)
        else:
            st.info("Données DeviceType non disponibles")
            
    
    # === SECTION 5: ANALYSE DES MONTANTS ===
    st.markdown("### 💰 Distribution des Montants de Transaction")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Distribution des montants
        fig_amount = utils.create_fraud_distribution_chart(df)
        st.plotly_chart(fig_amount, use_container_width=True)
    
    with col2:
        # Statistiques des montants
        if 'TransactionAmt' in df.columns:
            fraud_amounts = df[df['isFraud'] == 1]['TransactionAmt']
            legit_amounts = df[df['isFraud'] == 0]['TransactionAmt']
            
            st.markdown("#### 📊 Statistiques des Montants de Transaction")
            st.markdown(f"**Montant Moyen des Fraudes:** ${fraud_amounts.mean():,.2f}")
            st.markdown(f"**Montant Moyen des Transactions Légitimes:** ${legit_amounts.mean():,.2f}")
            st.markdown(f"**Montant Max. des Fraudes:** ${fraud_amounts.max():,.2f}")
            st.markdown(f"**Montant Max. des Transactions Légitimes:** ${legit_amounts.max():,.2f}")
        else:
            st.info("Données TransactionAmt non disponibles")
    st.markdown("---")
    # === SECTION 6: ANALYSE DES ZONES ===  
    st.markdown("### 🌍 Analyse des Zones Géographiques")
    if 'addr2' in df.columns:
        fig_zone = utils.create_fraud_rate_map(df)
        st.plotly_chart(fig_zone, use_container_width=True)
    else:
        st.info("Données addr2 non disponibles")
    st.markdown("---")
    # === SECTION 7: ANALYSE DES FRAUDES PAR TYPE DE CARTE ===
    st.markdown("### 💳 Analyse des Fraudes par Type de Carte")
    if 'card4' in df.columns:
        card_fraud_analysis = df.groupby('card4')['isFraud'].agg(['count', 'sum']).reset_index()
        card_fraud_analysis.columns = ['card_type', 'total', 'fraud_count']
        card_fraud_analysis['fraud_rate'] = (card_fraud_analysis['fraud_count'] / card_fraud_analysis['total'] * 100).round(2)
        
        fig_card = px.pie(
            card_fraud_analysis,
            values='fraud_count',
            names='card_type',
            title="Répartition des Fraudes par Type de Carte",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig_card, use_container_width=True)
    else:
        st.info("Données card4 non disponibles")
    st.markdown("---")
    # === SECTION 8: ANALYSE DES PRODUITS ===
    st.markdown("### 📦 Analyse des Produits")
    if 'ProductCD' in df.columns:
        product_fraud_analysis = df.groupby('ProductCD')['isFraud'].agg(['count', 'sum']).reset_index()
        product_fraud_analysis.columns = ['product', 'total', 'fraud_count']
        product_fraud_analysis['fraud_rate'] = (product_fraud_analysis['fraud_count'] / product_fraud_analysis['total'] * 100).round(2)
        
        fig_product = px.bar(
            product_fraud_analysis,
            x='product',
            y='fraud_rate',
            color='fraud_rate',
            color_continuous_scale=px.colors.sequential.Viridis,
            title="Taux de Fraude par Type de Produit",
            labels={'product': 'Type de Produit', 'fraud_rate': 'Taux de Fraude (%)'}
        )
        fig_product.update_layout(template='plotly_white')
        st.plotly_chart(fig_product, use_container_width=True)
    else:
        st.info("Données ProductCD non disponibles")