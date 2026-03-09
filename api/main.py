from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from src.predict import calculate_lags

templates = Jinja2Templates(directory="web/templates")

app = FastAPI()

app.mount("/static", StaticFiles(directory="web/static"), name="static")

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict-form")
def predict_form(request: Request, date: str = Form(...), temperature: float = Form(...)):
    try:
        prediction = calculate_lags(date, temperature)
        return templates.TemplateResponse("index.html", {"request": request, "prediction": prediction})
    except Exception as e:
        return templates.TemplateResponse("index.html", {"request": request, "error": str(e)})