from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

templates = Jinja2Templates(directory="web/templates")

app = FastAPI()

app.mount("/static", StaticFiles(directory="web/static"), name="static")

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict-form")
def predict_form(request: Request, date: str):
    # Here you would add your prediction logic using the input data
    # For demonstration, we'll just return a dummy prediction
    result = f"Predicted electricity consumption for {date}: 1234 kWh"
    
    return templates.TemplateResponse("index.html", {"request": request, "result": result})