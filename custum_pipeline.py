# fraud_preprocessing_pipeline.py

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from typing import Dict, List, Optional, Union




class DataTypeConverter:
    """Classe pour gérer la conversion des types de données à partir des variables originales"""
    
    @staticmethod
    def get_variable_types() -> Dict[str, Dict[str, str]]:
        """
        Définit les types de variables originales selon les transformations à appliquer dans le pipeline
        """
        return {
            # Variables quantitatives
            'quantitative': {
                'C1': 'float64', 'C2': 'float64', 'C4': 'float64', 'C6': 'float64',
                'C9': 'float64', 'C12': 'float64', 'C13': 'float64',
                'D3': 'float64', 'D5': 'float64',
                'V10': 'float64', 'V12': 'float64', 'V36': 'float64',
                'V49': 'float64', 'V50': 'float64', 'V53': 'float64', 'V61': 'float64',
                'V62': 'float64', 'V75': 'float64', 'V91': 'float64', 'V96': 'float64',
                'V283': 'float64', 'V285': 'float64',
                'TransactionAmt': 'float64'
            },
            
            # Variables catégorielles originales (non encore encodées)
            'categorical': {
                'id_01': 'object', 'id_05': 'object', 'id_06': 'object', 'id_11': 'object',
                'id_12': 'object', 'id_13': 'object', 'id_15': 'object',
                'id_19': 'object', 'id_20': 'object', 'id_31': 'object', 'id_38': 'object',
                'ProductCD': 'object', 'card1': 'object', 'card2': 'object', 'card5': 'object',
                'card6': 'object', 'addr1': 'object',
                'M3': 'object', 'M4': 'object', 'M6': 'object',
                'DeviceType': 'object', 'R_emaildomain': 'object', 'P_emaildomain': 'object'
            }
        }

    
    @staticmethod
    def convert_dtypes(df: pd.DataFrame) -> pd.DataFrame:
        """Convertit les types de données selon la classification"""
        df_converted = df.copy()
        type_mapping = DataTypeConverter.get_variable_types()
        
        # Conversion des variables quantitatives
        for col, dtype in type_mapping['quantitative'].items():
            if col in df_converted.columns:
                df_converted[col] = pd.to_numeric(df_converted[col], errors='coerce').astype(dtype)
        
        # Conversion des variables catégorielles
        for col, dtype in type_mapping['categorical'].items():
            if col in df_converted.columns:
                df_converted[col] = df_converted[col].astype(dtype)
        
        return df_converted


from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Encodeur de fréquence basé sur une liste explicite de variables."""

    def __init__(self):
        self.frequency_maps_ = {}   # dictionnaire : {col_name: {val: freq}}
        self.variables = None

    def set_variables(self, variables: list):
        """Définit explicitement les colonnes à encoder"""
        if not isinstance(variables, list):
            raise ValueError("❌ set_variables() attend une liste de chaînes.")
        self.variables = variables

    def fit(self, X: pd.DataFrame, y=None):
        if self.variables is None:
            self.variables = X.columns.tolist()
        self.frequency_maps_ = {}  # reset sécurisé

        for col in self.variables:
            if col in X.columns:
                freq = X[col].value_counts(normalize=True)
                self.frequency_maps_[col] = freq.to_dict()
            else:
                raise ValueError(f"❌ Colonne '{col}' non trouvée dans X.")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "frequency_maps_") or not isinstance(self.frequency_maps_, dict):
            raise ValueError("❌ frequency_maps_ doit être un dictionnaire")

        X_transformed = X.copy()

        for col in self.variables:
            if col in self.frequency_maps_:
                X_transformed[f"{col}_freq"] = X[col].map(self.frequency_maps_[col]).fillna(0)
            else:
                raise ValueError(f"❌ Pas de fréquence disponible pour la colonne '{col}'.")

        return X_transformed
    
    
class CardAddrCombiner(BaseEstimator, TransformerMixin):
    """Crée une colonne combinée card1 + addr1"""
    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_ = X.copy()
        if 'card1' in X_.columns and 'addr1' in X_.columns:
            X_['card1_addr1'] = X_['card1'].astype(str) + '_' + X_['addr1'].astype(str)
        return X_


class EmailDomainExtractor(BaseEstimator, TransformerMixin):
    """Extracteur du fournisseur principal d'email"""
    
    def __init__(self, email_columns: List[str]):
        self.email_columns = email_columns
    
    def fit(self, X: pd.DataFrame, y=None):
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_transformed = X.copy()
        
        for col in self.email_columns:
            if col in X.columns:
                # Extraction du fournisseur principal (avant le premier point)
                new_col = f"{col}_1"
                X_transformed[new_col] = (
                    X_transformed[col]
                    .fillna('')
                    .apply(lambda x: x.split('.')[0] if x else np.nan)
                    .replace({'': np.nan})
                )
        
        return X_transformed


class MontantTransformer(BaseEstimator, TransformerMixin):
    """Transformateur pour les variables de montant"""
    
    def __init__(self, amount_col: str = 'TransactionAmt'):
        self.amount_col = amount_col
        self.median_value_ = None
        
    def fit(self, X: pd.DataFrame, y=None):
        if self.amount_col in X.columns:
            median = X[self.amount_col].median()
            if pd.isna(median):
                raise ValueError(f"La médiane de {self.amount_col} est NaN ! Vérifiez les données.")
            self.median_value_ = median
        else:
            raise ValueError(f"{self.amount_col} n'est pas présent dans les colonnes du DataFrame")
        return self

    
    
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_transformed = X.copy()
        
        if self.amount_col in X.columns:
            # Imputation par la médiane
            amount_col_imputed = X_transformed[self.amount_col].fillna(self.median_value_)
            
            # Transformation log
            X_transformed['Log_Montant'] = np.log1p(amount_col_imputed)
            
            # Comptage des décimales
            X_transformed['Cents_Montant'] = amount_col_imputed.apply(
                lambda x: x - np.floor(x) 
            )
        
        return X_transformed



class FraudPreprocessingPipeline:
    """Pipeline principal de preprocessing"""
    
    def __init__(self):
        self.frequency_encoder = FrequencyEncoder()
        self.email_extractor = EmailDomainExtractor(['R_emaildomain', 'P_emaildomain'])
        self.montant_transformer = MontantTransformer()
        self.card_addr_combiner = CardAddrCombiner()  # <- ajouté ici

        self.imputation_values_ = {}
        self.is_fitted = False
        
        # Stockage des valeurs d'imputation calculées sur le train
        self.imputation_values_ = {}
        self.is_fitted = False
    
    
    def fit(self, X_train: pd.DataFrame, y_train=None):
        """Ajuste les transformateurs internes sur les données d'entraînement"""

        self.is_fitted = False  

        try:
            # Étape 0 : conversion des types
            X_processed = DataTypeConverter.convert_dtypes(X_train)

            # Étape 1 : imputation des valeurs manquantes
            self._calculate_imputation_values(X_processed)

            # Étape 2 : ajustement des transformateurs
            self.email_extractor.fit(X_processed)
            self.montant_transformer.fit(X_processed)
           

            # Étape 3 : application des transformations de base
            X_processed = self._apply_base_transformations(X_processed)

            # Étape 4 : encodage par fréquence
            # préparer les variables à encoder en fréquence
            freq_vars = self._prepare_frequency_variables(X_processed)  
            self.frequency_encoder.set_variables(freq_vars)             
            self.frequency_encoder.fit(X_processed)                     


            # Tout s'est bien passé
            self.is_fitted = True
            return self

        except Exception as e:
            # Sécurité : on garde une trace claire si échec
            self.is_fitted = False
            raise e



    
    def _calculate_imputation_values(self, X: pd.DataFrame):
        """Calcule les valeurs d'imputation sur les données d'entraînement"""
        print("=== CALCUL DES VALEURS D'IMPUTATION SUR LE TRAIN ===")
        
        # Variables C (moyenne pour imputation)
        c_vars = ['C1', 'C2', 'C4', 'C6', 'C9', 'C12', 'C13']
        for var in c_vars:
            if var in X.columns:
                mean_val = X[var].mean()
                self.imputation_values_[f"{var}_mean"] = mean_val
                na_count = X[var].isna().sum()
                print(f"  {var}: {na_count} NA -> imputation par moyenne = {mean_val:.4f}")
        
        # Variables card (mode pour imputation)
        card_vars = ['card1', 'card2', 'card5']
        for var in card_vars:
            if var in X.columns:
                mode_val = X[var].mode().iloc[0] if not X[var].mode().empty else 'undefined'
                self.imputation_values_[f"{var}_mode"] = mode_val
                na_count = X[var].isna().sum()
                print(f"  {var}: {na_count} NA -> imputation par mode = {mode_val}")
        
        # ProductCD (mode pour imputation)
        if 'ProductCD' in X.columns:
            mode_val = X['ProductCD'].mode().iloc[0] if not X['ProductCD'].mode().empty else 'W'
            self.imputation_values_['ProductCD_mode'] = mode_val
            na_count = X['ProductCD'].isna().sum()
            print(f"  ProductCD: {na_count} NA -> imputation par mode = {mode_val}")
        
        # Variables avec imputation fixe (pas de calcul nécessaire)
        fixed_imputations = {
            'V_vars': ('V36,V52,V53,V54,V74,V87,V96,V97,V222,V280,V283', -1),
            'D_vars': ('D3,D5', 999),
            'id_vars': ('id_01,id_05,id_06,id_12,id_13,id_15,id_19,id_20,id_31,id_38', 'undefined'),
            'M_vars': ('M4,M6', 'undefined'),
            'other_vars': ('addr1,DeviceType', 'undefined/Unknown')
        }
        
        for group, (vars_str, impute_val) in fixed_imputations.items():
            vars_list = vars_str.split(',')
            for var in vars_list:
                if var in X.columns:
                    na_count = X[var].isna().sum()
                    if na_count > 0:
                        actual_val = 'Unknown' if var == 'DeviceType' else ('undefined' if group in ['id_vars', 'M_vars'] else impute_val)
                        print(f"  {var}: {na_count} NA -> imputation fixe = {actual_val}")
        
        print("="*50)
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transforme les données selon le pipeline"""
        if not self.is_fitted:
            raise ValueError("Le pipeline doit être ajusté avant la transformation")
        
        # Application des transformations de base
        X_processed = self._apply_base_transformations(X)
        
        freq_vars = self._prepare_frequency_variables(X_processed)
        freq_encoded = self.frequency_encoder.transform(X_processed[freq_vars])

        for col in freq_encoded.columns:
            X_processed[col] = freq_encoded[col]        
            # Sélection des variables finales
            final_vars = self._get_final_variables()
            available_vars = [var for var in final_vars if var in X_processed.columns]
            
        return X_processed[available_vars]
    
    def fit_transform(self, X_train: pd.DataFrame, y_train=None) -> pd.DataFrame:
        """Ajuste et transforme les données d'entraînement"""
        return self.fit(X_train, y_train).transform(X_train)
    
    def _apply_base_transformations(self, X: pd.DataFrame) -> pd.DataFrame:
        """Applique les transformations de base avec les valeurs d'imputation"""
        X_transformed = X.copy()

        if not hasattr(self, 'imputation_values_'):
            X_transformed = DataTypeConverter.convert_dtypes(X_transformed)

        print("=== APPLICATION DES IMPUTATIONS ===")

        self._impute_quantitative_variables(X_transformed)
        self._impute_categorical_variables(X_transformed)
        X_transformed = self.email_extractor.transform(X_transformed)
        X_transformed = self.montant_transformer.transform(X_transformed)
        X_transformed = self.card_addr_combiner.transform(X_transformed)  # <- ligne réactivée

        print("=" * 40)
        return X_transformed
        
        
    
    def _impute_quantitative_variables(self, X: pd.DataFrame):
        c_vars = ['C1', 'C2', 'C4', 'C6', 'C9', 'C12', 'C13']
        for var in c_vars:
            if var in X.columns:
                mean_val = self.imputation_values_.get(f"{var}_mean", X[var].mean()) if self.is_fitted else X[var].mean()
                X[var] = X[var].fillna(mean_val)

        v_vars = ['V10', 'V12', 'V36', 'V49', 'V50', 'V53', 'V61', 'V62', 'V75', 'V91', 'V96', 'V283', 'V285']
        for var in v_vars:
            if var in X.columns:
                X[var] = X[var].fillna(-1)

        d_vars = ['D3', 'D5']
        for var in d_vars:
            if var in X.columns:
                X[var] = X[var].fillna(999)

        # Variables D (imputation par 999)
        d_vars = ['D3', 'D5']
        for var in d_vars:
            if var in X.columns:
                na_count_before = X[var].isna().sum()
                X[var] = X[var].fillna(999)
                if na_count_before > 0:
                    print(f"  {var}: {na_count_before} NA imputés par 999")

    def _impute_categorical_variables(self, X: pd.DataFrame):
        id_vars = ['id_01', 'id_05', 'id_06', 'id_11', 'id_12', 'id_13', 'id_15', 'id_19', 'id_20', 'id_31', 'id_38']
        for var in id_vars:
            if var in X.columns:
                X[var] = X[var].fillna('undefined')

        m_vars = ['M3', 'M4', 'M6']
        for var in m_vars:
            if var in X.columns:
                X[var] = X[var].fillna('undefined')

        card_vars = ['card1', 'card2', 'card5', 'card6']
        for var in card_vars:
            if var in X.columns:
                mode_val = self.imputation_values_.get(f"{var}_mode", 'undefined') if self.is_fitted else X[var].mode().iloc[0]
                X[var] = X[var].fillna(mode_val)

        for var in ['addr1', 'card1_addr1']:
            if var in X.columns:
                X[var] = X[var].fillna('undefined')

        if 'ProductCD' in X.columns:
            mode_val = self.imputation_values_.get('ProductCD_mode', 'W') if self.is_fitted else X['ProductCD'].mode().iloc[0]
            X['ProductCD'] = X['ProductCD'].fillna(mode_val)

        if 'DeviceType' in X.columns:
            X['DeviceType'] = X['DeviceType'].fillna('Unknown')


    def _prepare_frequency_variables(self, X: pd.DataFrame) -> list:
        """Prépare la liste des variables à encoder en fréquence"""
        freq_columns = []

        id_vars = ['id_01', 'id_05', 'id_06', 'id_11', 'id_12', 'id_13', 'id_15', 'id_19', 'id_20', 'id_31', 'id_38']
        freq_columns.extend([var for var in id_vars if var in X.columns])

        m_vars = ['M3', 'M4', 'M6']  # <- ajout de M3
        freq_columns.extend([var for var in m_vars if var in X.columns])

        email_vars = ['R_emaildomain_1', 'P_emaildomain_1']
        freq_columns.extend([var for var in email_vars if var in X.columns])

        card_vars = ['card1', 'card2', 'card5', 'card6']  # <- ajout de card6
        freq_columns.extend([var for var in card_vars if var in X.columns])

        addr_vars = ['addr1', 'card1_addr1']
        freq_columns.extend([var for var in addr_vars if var in X.columns])

        other_vars = ['ProductCD', 'DeviceType']
        freq_columns.extend([var for var in other_vars if var in X.columns])

        print(" Colonnes à encoder par fréquence :", freq_columns)
        return freq_columns


    def _get_final_variables(self) -> List[str]:
        return [
            'C2', 'id_06_freq', 'V50', 'id_20_freq', 'V61', 'V10', 'V53',
            'id_13_freq', 'id_38_freq', 'C4', 'V96', 'id_31_freq', 'C9',
            'id_05_freq', 'id_01_freq', 'id_11_freq', 'id_19_freq', 'id_12_freq',
            'id_15_freq', 'V49', 'C12', 'C13', 'V285', 'V91', 'V75', 'V283', 'C1',
            'C6', 'V62', 'V12', 'V36', 'D3', 'D5', 'Log_Montant', 'Cents_Montant',
            'ProductCD_freq', 'card6_freq', 'M3_freq', 'M4_freq', 'M6_freq',
            'DeviceType_freq', 'R_emaildomain_1_freq', 'P_emaildomain_1_freq',
            'card1_addr1_freq', 'card1_freq', 'card2_freq', 'card5_freq', 'addr1_freq'
        ]


    
    
    def get_sklearn_pipeline(self):
        """
        Version simplifiée pour visualisation dans un notebook
        (représente les grandes étapes du pipeline).
        """
        preprocessing_steps = Pipeline([
            ('email_extraction', FunctionTransformer(self.email_extractor.transform)),
            ('montant_transformation', FunctionTransformer(self.montant_transformer.transform)),
        ])

        diagram_pipeline = Pipeline([
            ('type_conversion', FunctionTransformer(DataTypeConverter.convert_dtypes)),
            ('base_transformations', preprocessing_steps),
            ('frequency_encoding', FunctionTransformer(self.frequency_encoder.transform))
        ])
        
        return diagram_pipeline

def extract_required_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extrait uniquement les variables nécessaires de la base de test
    et affiche la classification des types
    """
    required_vars = [
        # Variables originales nécessaires pour les transformations
        'id_01', 'id_05', 'id_06', 'id_12', 'id_13', 'id_15', 'id_19', 'id_20', 'id_31', 'id_38',
        'C1', 'C2', 'C4', 'C6', 'C9', 'C12', 'C13',
        'V36', 'V49', 'V50', 'V53', 'V61', 'V62', 'V10', 'V12', 'V75', 'V91', 'V96', 'V285', 'V283',
        'D3', 'D5',
        'Log_Montant', 'Cents_Montant',
        'M3', 'M4', 'M6',
        'R_emaildomain_1', 'P_emaildomain_1',
        'card1', 'card2', 'card5',
        'addr1', 'card1_addr1',
        'ProductCD', 'DeviceType'
    ]

    available_vars = [var for var in required_vars if var in df.columns]
    missing_vars = [var for var in required_vars if var not in df.columns]

    print("="*60)
    print(f"EXTRACTION DES VARIABLES NÉCESSAIRES")
    print("="*60)
    print(f"Variables trouvées: {len(available_vars)}/{len(required_vars)}")

    if missing_vars:
        print(f"\n Variables manquantes: {missing_vars}")

    type_mapping = DataTypeConverter.get_variable_types()

    print(f"\n CLASSIFICATION DES TYPES DE VARIABLES:")
    print("-" * 40)

    print(" VARIABLES QUANTITATIVES (imputation numérique):")
    quant_found = 0
    for var in available_vars:
        if var in type_mapping['quantitative']:
            impute_method = ""
            if var.startswith('C'):
                impute_method = "→ imputation par moyenne"
            elif var.startswith('V'):
                impute_method = "→ imputation par -1"
            elif var.startswith('D'):
                impute_method = "→ imputation par 999"
            elif var == 'Log_Montant' or var == 'Cents_Montant':
                impute_method = "→ dérivé de TransactionAmt"

            print(f"  • {var}: {df[var].dtype} → {type_mapping['quantitative'].get(var, 'float64')} {impute_method}")
            quant_found += 1

    print(f"\n VARIABLES CATÉGORIELLES (imputation par modalité):")
    cat_found = 0
    for var in available_vars:
        if var in type_mapping['categorical']:
            impute_method = ""
            if var.startswith('id_') or var.startswith('M'):
                impute_method = "→ imputation par 'undefined'"
            elif var.startswith('card'):
                impute_method = "→ imputation par mode"
            elif var == 'ProductCD':
                impute_method = "→ imputation par mode"
            elif var == 'DeviceType':
                impute_method = "→ imputation par 'Unknown'"
            elif var in ['addr1', 'R_emaildomain_1', 'P_emaildomain_1']:
                impute_method = "→ imputation par 'undefined'"

            print(f"  • {var}: {df[var].dtype} → {type_mapping['categorical'].get(var, 'object')} {impute_method}")
            cat_found += 1

    print(f"\n RÉSUMÉ:")
    print(f"  • Variables quantitatives: {quant_found}")
    print(f"  • Variables catégorielles: {cat_found}")
    print(f"  • Total: {quant_found + cat_found}")
    print("="*60)

    return df[available_vars].copy()



#
if __name__ == "__main__":
    
    
    
    
    pass