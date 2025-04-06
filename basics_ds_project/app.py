from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import os
import numpy as np
import pandas as pd
from typing import Optional
import uvicorn
from src.datascience.pipeline.prediction_pipeline import PredictionPipeline

app = FastAPI()  # initializing a FastAPI app

# Setup templates
templates = Jinja2Templates(directory="templates")

# Mount static files if you have any
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except:
    pass

@app.get("/", response_class=HTMLResponse)  # route to display the home page
async def home_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/train")  # route to train the pipeline
async def training():
    os.system("python main.py")
    return {"message": "Training Successful!"}

@app.post("/predict", response_class=HTMLResponse)  # route to show the predictions in a web UI
async def predict(
    request: Request,
    fixed_acidity: float = Form(...),
    volatile_acidity: float = Form(...),
    citric_acid: float = Form(...),
    residual_sugar: float = Form(...),
    chlorides: float = Form(...),
    free_sulfur_dioxide: float = Form(...),
    total_sulfur_dioxide: float = Form(...),
    density: float = Form(...),
    pH: float = Form(...),
    sulphates: float = Form(...),
    alcohol: float = Form(...)
):
    try:
        data = [fixed_acidity, volatile_acidity, citric_acid, residual_sugar, 
                chlorides, free_sulfur_dioxide, total_sulfur_dioxide, 
                density, pH, sulphates, alcohol]
        data = np.array(data).reshape(1, 11)
        
        obj = PredictionPipeline()
        predict = obj.predict(data)

        return templates.TemplateResponse("results.html", {"request": request, "prediction": str(predict)})

    except Exception as e:
        print('The Exception message is: ', e)
        return templates.TemplateResponse("index.html", {"request": request, "error": "Something went wrong"})

@app.get("/predict", response_class=HTMLResponse)
async def predict_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8001, reload=True)