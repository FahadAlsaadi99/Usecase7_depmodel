from typing import Optional, Union
from fastapi import FastAPI, HTTPException
import joblib
from pydantic import BaseModel

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Welcome To my model"}

@app.get("/items/{item_id}")
async def read_item(item_id: int):
    return {"item_id": item_id}


model1 = joblib.load('DBSCAN_model_GK.joblib')
model2 = joblib.load('DBSCAN_model_def.joblib')
model3 = joblib.load('DBSCAN_model_mid.joblib')
model4 = joblib.load('DBSCAN_model_att.joblib')

scaler1 = joblib.load('scaler_GK.joblib')
scaler2 = joblib.load('scaler_def.joblib')
scaler3 = joblib.load('scaler_mid.joblib')
scaler4 = joblib.load('scaler_att.joblib')


class InputFeaturesModel1(BaseModel):
    appearance: str
    minutes_played: int
    games_injured: int
    award: str
    highest_value: str
    League_full_Premier_League: bool

class InputFeaturesModel2(BaseModel):
    appearance: str
    minutes_played: int
    games_injured: int
    award: str
    highest_value: str
    League_full_Premier_League: bool
    League_full_Serie_A: bool

class InputFeaturesModel3(BaseModel):
    appearance: str
    minutes_played: int
    games_injured: int
    award: str
    highest_value: str
    League_full_Premier_League: bool
    Premier_Soccer_League: bool

class InputFeaturesModel4(BaseModel):
    appearance: str
    minutes_played: int
    games_injured: int
    award: str
    highest_value: str
    League_full_Premier_League: bool
    Premier_Soccer_League: bool

@app.post("/predict")
async def predict(model_type: str, input_features: Union[InputFeaturesModel1, InputFeaturesModel2, InputFeaturesModel3, InputFeaturesModel4]):
    if model_type == 'model1':
        model = model1
        scaler = scaler1
    elif model_type == 'model2':
        model = model2
        scaler = scaler2
    elif model_type == 'model3':
        model = model3
        scaler = scaler3
    elif model_type == 'model4':
        model = model4
        scaler = scaler4
    else:
        raise HTTPException(status_code=400, detail="Invalid model_type")

    features = None
    if isinstance(input_features, InputFeaturesModel1):
        features = [
            input_features.appearance,
            input_features.minutes_played,
            input_features.games_injured,
            input_features.award,
            input_features.highest_value,
            input_features.League_full_Premier_League
        ]
    elif isinstance(input_features, InputFeaturesModel2):
        features = [
            input_features.appearance,
            input_features.minutes_played,
            input_features.games_injured,
            input_features.award,
            input_features.highest_value,
            input_features.League_full_Premier_League,
            input_features.League_full_Serie_A
        ]
    elif isinstance(input_features, InputFeaturesModel3):
        features = [
            input_features.appearance,
            input_features.minutes_played,
            input_features.games_injured,
            input_features.award,
            input_features.highest_value,
            input_features.League_full_Premier_League,
            input_features.Premier_Soccer_League
        ]
    elif isinstance(input_features, InputFeaturesModel4):
        features = [
            input_features.appearance,
            input_features.minutes_played,
            input_features.games_injured,
            input_features.award,
            input_features.highest_value,
            input_features.League_full_Premier_League,
            input_features.Premier_Soccer_League
        ]

    if features is None:
        raise HTTPException(status_code=400, detail="Invalid input features")

    scaled_features = scaler.transform([features])
    y_pred = model.predict(scaled_features)
    
    return {"pred": y_pred.tolist()[0]}
