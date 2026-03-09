# Electricity Price Forecasting (Next 24 Hours)

## Live Demo
https://electricity-forecast-328829738430.europe-west1.run.app/

## Project Overview

This project predicts the **electricity price for the next 24 hours** based on historical electricity market data in the PJM East region. The system processes historical data, trains a machine learning model, and exposes predictions through a web interface.

Users can input a **date and time**, and the system will return the predicted electricity price for that moment.

The application is deployed as a containerized service and can be accessed through a web interface powered by a backend API.

The prediction horizon is **strictly limited to 24 hours after the last available training timestamp (31/12/2027)**.

---

# Dataset and Data Processing

## Dataset

The dataset contains historical electricity market information including timestamps and electricity prices.

Typical structure:

| Timestamp | Price |
|-----------|------|
| 2015-01-01 00:00 | ... |
| 2015-01-01 01:00 | ... |

The data is organized at **hourly resolution**, which makes it suitable for time-series forecasting.

American Electric Power (AEP): estimated energy consumption in Megawatts (MW)

Commonwealth Edison (ComEd): estimated energy consumption in Megawatts (MW)

The Dayton Power and Light Company: estimated energy consumption in Megawatts (MW)

Duke Energy Ohio/Kentucky (DEOK): estimated energy consumption in Megawatts (MW)

Dominion Virginia Power (DOM): estimated energy consumption in Megawatts (MW)

Duquesne Light Co. (DUQ): estimated energy consumption in Megawatts (MW)

East Kentucky Power Cooperative (EKPC): estimated energy consumption in Megawatts (MW)

FirstEnergy (FE): estimated energy consumption in Megawatts (MW)

Northern Illinois Hub (NI): estimated energy consumption in Megawatts (MW)

**PJM East Region: 2001-2018 (PJME): estimated energy consumption in Megawatts (MW)**

PJM West Region: 2001-2018 (PJMW): estimated energy consumption in Megawatts (MW)

PJM Load Combined: 1998-2001 (PJM_Load): estimated energy consumption in Megawatts (MW)

est_hourly.paruqet: Combined All Regions Load in Megawatts (MW): American Electric Power (AEP), Commonwealth Edison (ComEd), Dayton Power and Light Company

pjm_hourly_est: Combined All Regions Load: American Electric Power (AEP), Commonwealth Edison (ComEd), Dayton Power and Light Company


---

## Data Preprocessing

Before training the model, the data undergoes several preprocessing steps:

### 1. Timestamp conversion

The timestamp column is converted to a proper datetime format to allow time-based feature extraction.

### 2. Feature engineering

From the timestamp we extract several useful predictors:

- Hour of the day
- Day of the week
- Month
- Day of the year

These features allow the model to capture **seasonal and daily electricity consumption patterns**.

### 3. Normalization

Numerical features are scaled using a **StandardScaler** to improve model convergence and performance.

### 4. Train/Test split

The dataset is split chronologically to preserve time order:

- Training data: historical observations
- Test data: most recent observations

This prevents **data leakage**, which is critical in time-series forecasting.


## Exploratory Data Analysis

During the exploratory analysis, some **extreme low demand values** appeared in the dataset.

After investigating these anomalies, they were identified as **real demand drops during Hurricane Sandy in 2012**.

Since these values correspond to real-world events and represent legitimate system behavior, they were **kept in the dataset** instead of being removed as outliers. Removing them would artificially smooth the series and reduce the model’s ability to handle extreme scenarios.


## Time Series Characteristics

The electricity demand series presents several important temporal patterns:

- **Very strong local dependency** (hour-to-hour autocorrelation)
- **Strong daily seasonality**
- **Moderate weekly seasonality**

These patterns are typical in electricity demand forecasting because consumption follows **human activity cycles** (daily routines and weekday/weekend differences).


## External Features

To improve forecasting performance, external variables related to weather were included.

### Temperature

Temperature data from **Philadelphia Airport** was used as a proxy for regional weather conditions.

Electricity demand is strongly influenced by temperature due to heating and cooling needs.

---

### Heating Degree Days (HDD)

Heating Degree Days measure how cold the weather is relative to a baseline temperature.

Higher HDD values indicate stronger heating demand.

---

### Cooling Degree Days (CDD)

Cooling Degree Days measure how hot the weather is relative to a baseline temperature.

Higher CDD values indicate stronger cooling demand.

---

### Why HDD and CDD?

The relationship between **temperature and electricity demand is non-linear**.

Demand tends to be:

- **high when temperatures are very cold**
- **high when temperatures are very hot**
- **low when temperatures are comfortable**

HDD and CDD transform this **U-shaped relationship** into features that are easier for machine learning models to interpret.


---

# Machine Learning Model

The forecasting model is implemented using **scikit-learn**.

### Model Type

Regression model trained to predict:

**Electricity price at a given timestamp**

Possible inputs include:

- Hour
- Day of week
- Month
- Other time-derived features

The model learns patterns such as:

- daily price cycles
- weekday vs weekend behavior
- seasonal patterns

The trained model is stored using: model.pkl

This allows the model to be loaded instantly by the API without retraining.

---

# Prediction Constraints

The system enforces a strict prediction window:

**Predictions are only allowed up to 24 hours after the final timestamp in the dataset (31/12/2027).**

This constraint prevents the model from making unrealistic long-term forecasts beyond the range it was designed for.

If a user submits a request outside this window, the API returns an error message.

---

# API

The backend API is implemented using **FastAPI**.

The API is responsible for:

- loading the trained model
- validating user input
- generating predictions
- returning results to the frontend

## Endpoint

### POST `/predict`

Input:
date: YYYY-MM-DD
hour: HH


Example request:
date=2027-12-31
hour=15


Response:
Predicted electricity price: XX €/MWh


The API also validates the input range to ensure the requested timestamp is within the allowed prediction window.

---

# Web Interface

The frontend is a simple web interface built with **HTML and CSS**.

The interface allows users to:

1. Select a date
2. Select an hour
3. Request a prediction

The form sends the input to the API using a POST request and displays the predicted price.

The goal of the interface is to provide a **simple demonstration of the model in production**.

---

# Docker

The application is containerized using **Docker**.

Containerization ensures the application runs consistently across environments.

The Docker container includes:

- Python runtime
- API server
- trained model
- frontend files

Typical build command:

docker build -t electricity-forecast .


Run locally:

docker run -p 8080:8080 electricity-forecast


---

# Deployment

The application is deployed using **Google Cloud Run**.

Cloud Run allows running containerized applications without managing servers.

Deployment workflow:

1. Build the Docker image
2. Push the image to the container registry
3. Deploy the container to Cloud Run


Once deployed, the application is accessible through a public URL.

---

# Tech Stack

### Programming Language

- Python

### Machine Learning

- scikit-learn
- pandas
- numpy

### Backend

- FastAPI
- Uvicorn

### Frontend

- HTML
- CSS

### Infrastructure

- Docker
- Google Cloud Run
- Google Cloud Build

---

# Possible Future Improvements

Several extensions could improve the project:

- add more features (weather, demand, renewable production)
- implement more advanced time-series models
- compare multiple models
- add interactive visualizations
- implement automated retraining pipelines

---

# Purpose of the Project

This project demonstrates how to move a machine learning model **from training to production**, including:

- data preprocessing
- model training
- API development
- containerization
- cloud deployment

It showcases the full **end-to-end workflow of a data science application**.
