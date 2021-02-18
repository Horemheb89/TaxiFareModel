
# write some code for the API here
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import joblib

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)
@app.get("/")
def index():
    return {"greeting": "Hello world"}

@app.get("/predict_fare")
def predict(
            pickup_datetime,
            pickup_longitude,
            pickup_latitude,
            dropoff_longitude,
            dropoff_latitude,
            passenger_count):

    query_data = pd.DataFrame(dict(
            pickup_datetime= [pickup_datetime],
            pickup_longitude= [float(pickup_longitude)],
            pickup_latitude= [float(pickup_latitude)],
            dropoff_longitude= [float(dropoff_longitude)],
            dropoff_latitude= [float(dropoff_latitude)],
            passenger_count= [int(passenger_count)]
            ))

    model = joblib.load('model.joblib')
    prediction = float(model.predict(query_data)[0])
    return {
            "prediction": prediction
            }
