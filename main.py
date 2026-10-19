import joblib
from fastapi import FastAPI, HTTPException,APIRouter,UploadFile,File
import pandas as pd
from pydantic import BaseModel,Field
from io import BytesIO

app=FastAPI()

model=joblib.load('california_housing_model.joblib')
features=joblib.load("california_housing_features.joblib")

# input schema
class HouseFeatures(BaseModel):
    MedInc:float=Field(gt=0,description="Median income in block group")
    HouseAge:float=Field(gt=0,description="Median house age in block group")
    AveRooms:float=Field(gt=0,description="Average number of rooms per household")
    AveBedrms:float=Field(gt=0,description="Average number of bedrooms per household")
    Population:float=Field(gt=0,description="Block group population")
    AveOccup:float=Field(gt=0,description="Average number of household members")
    Latitude:float=Field(ge=32,le=43,description="Block group latitude")
    Longitude:float=Field(ge=-125,le=-114,description="Block group longitude")

# Home
@app.get("/")
def home():
    return{"message":"Welcome to california housing price predction API",
           "status":"Running",
           "endpoint":" send post request to predict price of house"
           }

@app.get("/Health")
def health():
    return{"status":"running",
           "model":"RandomForestRegressor",
           "features":features,
           "avg_error":"$ 39,000"}

# prediction endpoint
@app.post("/predict")
def predict(house:HouseFeatures):
    try:
        input_data=pd.DataFrame([{
            
            "MedInc":house.MedInc,
            "HouseAge":house.HouseAge,
            "AveRooms":house.AveRooms,
            "AveBedrms":house.AveBedrms,
            "Population":house.Population,
            "AveOccup":house.AveOccup,
            "Latitude":house.Latitude,
            "Longitude":house.Longitude
        }])

        # make variable name predicted to tell model
        predicted=model.predict(input_data)[0]
        price_USD=predicted*100000
        return{"predicted_price":f"${price_USD:,.0f}"}

    
    except Exception as e:
         raise HTTPException(
            status_code=500,
            detail=str(e))

