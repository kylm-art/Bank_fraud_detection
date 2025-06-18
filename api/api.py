from fastapi import FastAPI
import mlflow.pyfunc
import pandas as pd
from pydantic import BaseModel
from typing import List

app = FastAPI(title="Fraud Detection API")

# Charger le modèle MLflow
model_path = "runs:/6746767622b5417eb6985fb690b1b5d7/model"  

class Transaction(BaseModel):
    TransactionDT: float
    TransactionAMT: float
    ProductCD: str
    card1: str
    card2: str
    card3: str
    card4: str
    card5: str
    card6: str
    P_emaildomain: str
    addr1: str
    addr2: str
    # Ajoutez les autres champs nécessaires

class PredictionResult(BaseModel):
    transaction_id: int
    fraud_probability: float
    is_fraud: bool
    top_features: List[dict]

@app.post("/predict", response_model=PredictionResult)
async def predict(transaction: Transaction):
    # Convertir la transaction en DataFrame
    data = pd.DataFrame([transaction.dict()])
    
    # Prétraitement (identique à celui de l'entraînement)
    data = preprocess_data(data)
    
    # Prédiction
    proba = model.predict_proba(data)[0][1]
    prediction = proba > 0.5  # Seuil à ajuster
    
    # Features importantes (pour RandomForest)
    feature_importances = model.unwrap_python_model().feature_importances_
    features = sorted(zip(data.columns, feature_importances), key=lambda x: x[1], reverse=True)[:5]
    top_features = [{"feature": f[0], "importance": float(f[1])} for f in features]
    
    return {
        "transaction_id": 0,  # À remplacer par un ID réel
        "fraud_probability": float(proba),
        "is_fraud": bool(prediction),
        "top_features": top_features
    }

@app.post("/predict_batch")
async def predict_batch(file: UploadFile = File(...)):
    # Charger le fichier
    if file.filename.endswith('.csv'):
        data = pd.read_csv(file.file)
    elif file.filename.endswith('.parquet'):
        data = pd.read_parquet(file.file)
    else:
        raise HTTPException(400, "Format de fichier non supporté")
    
    # Prétraitement et prédiction
    data_processed = preprocess_data(data)
    predictions = model.predict_proba(data_processed)[:, 1]
    
    # Ajout des prédictions aux données
    data['fraud_probability'] = predictions
    data['is_fraud'] = predictions > 0.5
    
    return data.to_dict(orient='records')

def preprocess_data(df):
    # Implémentez le même prétraitement que dans train_model.py
    pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)