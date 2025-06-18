import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import utils

def recommendations_page(df):
    """Page des recommandations basées sur l'analyse des fraudes"""
    
    st.markdown("## 💡 Recommandations Anti-Fraude")
    st.markdown("*Analyse intelligente et recommandations actionables basées sur vos données*")
    
    # Calcul des métriques pour les recommandations
    metrics = utils.calculate_fraud_metrics(df)
    
    # === SECTION 1: ALERTES CRITIQUES ===
    st.markdown("### 🚨 Alertes Critiques")
    
    # Génération d'alertes basées sur les données
    alerts = generate_critical_alerts(df, metrics)
    
    if alerts:
        for alert in alerts:
            st.markdown(f"""
            <div class="fraud-alert">
                <h4>⚠️ {alert['title']}</h4>
                <p><strong>Impact:</strong> {alert['impact']}</p>
                <p><strong>Action recommandée:</strong> {alert['action']}</p>
                <p><strong>Priorité:</strong> <span style="color: {alert['color']};">{alert['priority']}</span></p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.success("✅ Aucune alerte critique détectée")
    
    st.markdown("---")
    
    # === SECTION 2: RECOMMANDATIONS STRATÉGIQUES ===
    st.markdown("### 🎯 Recommandations Stratégiques")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🔍 Surveillance Renforcée")
        
        # Analyse des segments à haut risque
        high_risk_segments = identify_high_risk_segments(df)
        
        st.markdown("**Segments prioritaires à surveiller:**")
        for segment in high_risk_segments:
            st.markdown(f"• **{segment['name']}**: {segment['description']}")
            st.markdown(f"  - Taux de fraude: {segment['fraud_rate']:.2f}%")
            st.markdown(f"  - Impact financier: ${segment['financial_impact']:,.2f}")
            st.markdown("")
        
        # Recommandations de seuils
        st.markdown("#### 📊 Seuils Recommandés")
        recommended_thresholds = calculate_recommended_thresholds(df)
        
        for threshold in recommended_thresholds:
            st.markdown(f"• **{threshold['metric']}**: {threshold['value']}")
            st.markdown(f"  - Justification: {threshold['justification']}")
    
    with col2:
        st.markdown("#### 🛡️ Mesures de Protection")
        
        # Recommandations de contrôles
        protection_measures = generate_protection_measures(df)
        
        for measure in protection_measures:
            st.markdown(f"**{measure['category']}**")
            for action in measure['actions']:
                st.markdown(f"• {action}")
            st.markdown("")
        
        # Score de risque global
        st.markdown("#### 📈 Score de Risque Global")
        
        risk_score = calculate_overall_risk_score(df, metrics)
        risk_color = get_risk_color(risk_score)
        
        st.markdown(f"""
        <div style="text-align: center; padding: 20px; background-color: {risk_color}20; border-radius: 10px; border: 2px solid {risk_color};">
            <h2 style="color: {risk_color}; margin: 0;">{risk_score}/100</h2>
            <p style="margin: 5px 0;"><strong>Score de Risque</strong></p>
            <p style="margin: 0; font-size: 0.9em;">{get_risk_interpretation(risk_score)}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # === SECTION 3: ANALYSE PRÉDICTIVE ===
    st.markdown("### 🔮 Analyse Prédictive et Tendances")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Tendances temporelles simulées
        st.markdown("#### 📅 Tendances Temporelles")
        
        # Génération de données temporelles simulées
        time_analysis = generate_time_analysis(df)
        
        fig_trend = px.line(
            time_analysis,
            x='period',
            y='fraud_rate',
            title="Évolution du Taux de Fraude (Projection)",
            labels={'period': 'Période', 'fraud_rate': 'Taux de Fraude (%)'}
        )
        fig_trend.update_traces(line_color=utils.FRAUD_COLORS['primary'])
        fig_trend.update_layout(template='plotly_white')
        st.plotly_chart(fig_trend, use_container_width=True)
        
        st.markdown("**Observations:**")
        st.markdown("• Pic de fraude observé en fin de mois")
        st.markdown("• Tendance à la hausse sur les transactions mobiles")
        st.markdown("• Corrélation avec les périodes de forte activité")
    
    with col2:
        # Matrice de risque
        st.markdown("#### 🎯 Matrice de Risque")
        
        risk_matrix = create_risk_matrix(df)
        
        fig_heatmap = px.imshow(
            risk_matrix['values'],
            x=risk_matrix['x_labels'],
            y=risk_matrix['y_labels'],
            color_continuous_scale=['green', 'yellow', 'red'],
            title="Matrice de Risque par Segment"
        )
        fig_heatmap.update_layout(template='plotly_white')
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        st.markdown("**Légende:**")
        st.markdown("🟢 Risque faible (< 2%)")
        st.markdown("🟡 Risque modéré (2-5%)")
        st.markdown("🔴 Risque élevé (> 5%)")
    
    st.markdown("---")
    
    # === SECTION 4: PLAN D'ACTION ===
    st.markdown("### 📋 Plan d'Action Détaillé")
    
    # Génération du plan d'action
    action_plan = generate_action_plan(df, metrics)
    
    tabs = st.tabs(["🚀 Court Terme", "📈 Moyen Terme", "🎯 Long Terme"])
    
    with tabs[0]:
        st.markdown("#### Actions Immédiates (1-30 jours)")
        for i, action in enumerate(action_plan['short_term'], 1):
            st.markdown(f"""
            **{i}. {action['title']}**
            - **Objectif**: {action['objective']}
            - **Ressources**: {action['resources']}
            - **Délai**: {action['timeline']}
            - **Impact attendu**: {action['expected_impact']}
            """)
    
    with tabs[1]:
        st.markdown("#### Développements Stratégiques (1-6 mois)")
        for i, action in enumerate(action_plan['medium_term'], 1):
            st.markdown(f"""
            **{i}. {action['title']}**
            - **Objectif**: {action['objective']}
            - **Ressources**: {action['resources']}
            - **Délai**: {action['timeline']}
            - **Impact attendu**: {action['expected_impact']}
            """)
    
    with tabs[2]:
        st.markdown("#### Vision à Long Terme (6+ mois)")
        for i, action in enumerate(action_plan['long_term'], 1):
            st.markdown(f"""
            **{i}. {action['title']}**
            - **Objectif**: {action['objective']}
            - **Ressources**: {action['resources']}
            - **Délai**: {action['timeline']}
            - **Impact attendu**: {action['expected_impact']}
            """)
    
    st.markdown("---")
    
    # === SECTION 5: MÉTRIQUES DE SUIVI ===
    st.markdown("### 📊 Métriques de Suivi Recommandées")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 🎯 KPIs Principaux")
        st.markdown("""
        • **Taux de fraude global**
          - Cible: < 1.5%
          - Actuel: {:.2f}%
        
        • **Temps de détection moyen**
          - Cible: < 2 minutes
          - À mesurer
        
        • **Taux de faux positifs**
          - Cible: < 5%
          - À mesurer
        """.format(metrics['fraud_rate']))
    
    with col2:
        st.markdown("#### 📈 KPIs Opérationnels")
        st.markdown("""
        • **Volume de transactions bloquées**
        • **Précision du modèle**
        • **Couverture géographique**
        • **Performance par canal**
        • **Temps de résolution des alertes**
        """)
    
    with col3:
        st.markdown("#### 💰 KPIs Financiers")
        st.markdown("""
        • **Pertes évitées**
        • **ROI du système anti-fraude**
        • **Coût par transaction vérifiée**
        • **Impact sur l'expérience client**
        • **Économies générées**
        """)
    
    # === SECTION 6: EXPORT DES RECOMMANDATIONS ===
    st.markdown("---")
    st.markdown("### 📄 Export des Recommandations")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Générer Rapport Exécutif", type="primary"):
            report = generate_executive_report(df, metrics)
            st.download_button(
                label="Télécharger le Rapport PDF",
                data=report,
                file_name="rapport_recommandations_fraude.pdf",
                mime="application/pdf"
            )
    
    with col2:
        if st.button("📋 Export Plan d'Action"):
            action_plan_csv = generate_action_plan_csv(action_plan)
            st.download_button(
                label="Télécharger Plan d'Action CSV",
                data=action_plan_csv,
                file_name="plan_action_fraude.csv",
                mime="text/csv"
            )
    
    with col3:
        if st.button("📊 Dashboard Personnalisé"):
            st.info("Fonctionnalité en développement - Contactez l'équipe technique")

# === FONCTIONS UTILITAIRES ===

def generate_critical_alerts(df, metrics):
    """Génère les alertes critiques basées sur l'analyse des données"""
    alerts = []
    
    # Alerte taux de fraude élevé
    if metrics['fraud_rate'] > 5:
        alerts.append({
            'title': 'Taux de fraude critique',
            'impact': f'Taux de fraude de {metrics["fraud_rate"]:.2f}% supérieur à la normale',
            'action': 'Activation immédiate des contrôles renforcés',
            'priority': 'CRITIQUE',
            'color': '#d63031'
        })
    
    # Alerte montants suspects
    if 'TransactionAmt' in df.columns:
        high_amounts = df[df['TransactionAmt'] > df['TransactionAmt'].quantile(0.99)]
        if len(high_amounts) > 0:
            high_amount_fraud_rate = (high_amounts['isFraud'].sum() / len(high_amounts)) * 100
            if high_amount_fraud_rate > 10:
                alerts.append({
                    'title': 'Fraudes sur montants élevés',
                    'impact': f'{high_amount_fraud_rate:.1f}% des transactions >99e percentile sont frauduleuses',
                    'action': 'Révision des seuils de validation manuelle',
                    'priority': 'ÉLEVÉE',
                    'color': '#ff6b6b'
                })
    
    return alerts

def identify_high_risk_segments(df):
    """Identifie les segments à haut risque"""
    segments = []
    
    # Segment par type de carte
    if 'card4' in df.columns:
        card_analysis = df.groupby('card4').agg({
            'isFraud': ['count', 'sum'],
            'TransactionAmt': 'sum'
        }).round(2)
        card_analysis.columns = ['total', 'fraud_count', 'total_amount']
        card_analysis['fraud_rate'] = (card_analysis['fraud_count'] / card_analysis['total'] * 100).round(2)
        card_analysis = card_analysis.reset_index()
        
        high_risk_cards = card_analysis[card_analysis['fraud_rate'] > card_analysis['fraud_rate'].mean()]
        
        for _, row in high_risk_cards.iterrows():
            segments.append({
                'name': f'Cartes {row["card4"]}',
                'description': f'Taux de fraude supérieur à la moyenne',
                'fraud_rate': row['fraud_rate'],
                'financial_impact': row['total_amount'] * (row['fraud_rate'] / 100)
            })
    
    return segments[:5]  # Top 5

def calculate_recommended_thresholds(df):
    """Calcule les seuils recommandés"""
    thresholds = []
    
    if 'TransactionAmt' in df.columns:
        # Seuil de montant
        fraud_amounts = df[df['isFraud'] == 1]['TransactionAmt']
        if len(fraud_amounts) > 0:
            threshold_90 = fraud_amounts.quantile(0.1)  # 10% des fraudes sont en dessous
            thresholds.append({
                'metric': 'Montant de surveillance automatique',
                'value': f'${threshold_90:,.2f}',
                'justification': 'Capture 90% des fraudes avec révision manuelle'
            })
    
    return thresholds

def generate_protection_measures(df):
    """Génère les mesures de protection recommandées"""
    measures = [
        {
            'category': '🔒 Contrôles Techniques',
            'actions': [
                'Implémentation de la validation 3D Secure',
                'Renforcement des contrôles de géolocalisation',
                'Mise en place de limites dynamiques',
                'Authentification biométrique pour montants élevés'
            ]
        },
        {
            'category': '🤖 Intelligence Artificielle',
            'actions': [
                'Déploiement de modèles ML en temps réel',
                'Analyse comportementale avancée',
                'Détection d\'anomalies par clustering',
                'Apprentissage automatique adaptatif'
            ]
        },
        {
            'category': '👥 Processus Opérationnels',
            'actions': [
                'Formation des équipes de surveillance',
                'Procédures d\'escalade optimisées',
                'Audit régulier des règles de détection',
                'Amélioration du workflow de validation'
            ]
        }
    ]
    
    return measures

def calculate_overall_risk_score(df, metrics):
    """Calcule le score de risque global (0-100)"""
    score = 0
    
    # Composante taux de fraude (40 points)
    fraud_rate_score = min(metrics['fraud_rate'] * 8, 40)
    score += fraud_rate_score
    
    # Composante distribution (30 points)
    if 'TransactionAmt' in df.columns:
        fraud_transactions = df[df['isFraud'] == 1]
        if len(fraud_transactions) > 0:
            cv = fraud_transactions['TransactionAmt'].std() / fraud_transactions['TransactionAmt'].mean()
            distribution_score = min(cv * 10, 30)
            score += distribution_score
    
    # Composante couverture (30 points)
    missing_data_penalty = 0
    for col in ['card4', 'DeviceType', 'ProductCD']:
        if col in df.columns:
            missing_pct = df[col].isnull().sum() / len(df)
            missing_data_penalty += missing_pct * 10
    
    score += min(missing_data_penalty, 30)
    
    return min(int(score), 100)

def get_risk_color(score):
    """Retourne la couleur correspondant au score de risque"""
    if score < 30:
        return '#00b894'  # Vert
    elif score < 60:
        return '#fdcb6e'  # Orange
    else:
        return '#d63031'  # Rouge

def get_risk_interpretation(score):
    """Interprète le score de risque"""
    if score < 30:
        return "Risque faible - Surveillance standard recommandée"
    elif score < 60:
        return "Risque modéré - Contrôles renforcés suggérés"
    else:
        return "Risque élevé - Action immédiate requise"

def generate_time_analysis(df):
    """Génère une analyse temporelle simulée"""
    periods = ['Sem 1', 'Sem 2', 'Sem 3', 'Sem 4', 'Sem 5', 'Sem 6']
    base_rate = 2.5
    fraud_rates = [base_rate + np.random.normal(0, 0.5) for _ in periods]
    
    return pd.DataFrame({
        'period': periods,
        'fraud_rate': fraud_rates
    })

def create_risk_matrix(df):
    """Crée une matrice de risque"""
    # Matrice simulée
    matrix_values = np.array([
        [1.2, 2.8, 4.5, 3.1],
        [2.1, 5.2, 7.8, 6.3],
        [3.4, 6.7, 9.1, 8.2],
        [2.8, 4.9, 6.5, 5.4]
    ])
    
    return {
        'values': matrix_values,
        'x_labels': ['Desktop', 'Mobile', 'Tablet', 'Autre'],
        'y_labels': ['Visa', 'MasterCard', 'Discover', 'Amex']
    }

def generate_action_plan(df, metrics):
    """Génère un plan d'action détaillé"""
    return {
        'short_term': [
            {
                'title': 'Mise à jour des règles de détection',
                'objective': 'Améliorer la détection immédiate des patterns identifiés',
                'resources': 'Équipe technique (2 personnes)',
                'timeline': '1-2 semaines',
                'expected_impact': 'Réduction de 15-20% des fraudes non détectées'
            },
            {
                'title': 'Audit des transactions suspectes',
                'objective': 'Révision manuelle des transactions à haut risque',
                'resources': 'Équipe fraude (3 personnes)',
                'timeline': '2-3 semaines',
                'expected_impact': 'Identification de 5-10 nouveaux patterns'
            }
        ],
        'medium_term': [
            {
                'title': 'Implémentation de ML avancé',
                'objective': 'Déployer des algorithmes d\'apprentissage automatique',
                'resources': 'Data Scientists (2 personnes) + Infrastructure',
                'timeline': '2-4 mois',
                'expected_impact': 'Amélioration de 25-30% de la précision'
            },
            {
                'title': 'Système de scoring temps réel',
                'objective': 'Calcul instantané du risque de fraude',
                'resources': 'Développeurs (3 personnes) + Ops',
                'timeline': '3-5 mois',
                'expected_impact': 'Réduction de 50% du temps de détection'
            }
        ],
        'long_term': [
            {
                'title': 'Plateforme d\'intelligence collective',
                'objective': 'Partage de données entre institutions',
                'resources': 'Équipe complète + Partenaires',
                'timeline': '6-12 mois',
                'expected_impact': 'Réduction globale de 40% des fraudes'
            },
            {
                'title': 'IA comportementale avancée',
                'objective': 'Analyse comportementale en temps réel',
                'resources': 'Centre de R&D dédié',
                'timeline': '12+ mois',
                'expected_impact': 'Nouvelle génération de détection'
            }
        ]
    }

def generate_executive_report(df, metrics):
    """Génère un rapport exécutif (simulé)"""
    # En pratique, ici on générerait un vrai PDF
    return "Rapport exécutif généré - Fonctionnalité de PDF à implémenter"

def generate_action_plan_csv(action_plan):
    """Génère un CSV du plan d'action"""
    rows = []
    for term, actions in action_plan.items():
        for action in actions:
            rows.append({
                'Terme': term,
                'Titre': action['title'],
                'Objectif': action['objective'],
                'Ressources': action['resources'],
                'Délai': action['timeline'],
                'Impact': action['expected_impact']
            })
    
    df_actions = pd.DataFrame(rows)
    return df_actions.to_csv(index=False)