# Industrial Safety Monitoring Dashboard

This project turns the provided incident dataset into a practical internship-ready dashboard for monitoring safety conditions inside a facility.

## What it does

- Loads and explores a synthetic industrial incident dataset
- Identifies risk trends from temperature, smoke, gas, sound, crowd, motion, and flame data
- Trains a machine learning model to classify incident types
- Shows predictions interactively from the Streamlit dashboard
- Presents charts, key metrics, and model performance insights

## Dataset

The project uses the included file at `data/incident_data.csv`.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL shown in the terminal.

## Model

The app uses a `RandomForestClassifier` with preprocessing to predict one of the incident classes such as:

- Safe / Normal Activity
- Minor Smoke
- Possible Gas Issue
- Crowd Disturbance
- Unusual Heat
- Heavy Smoke
- Severe Heat Emergency
- Critical Gas Leak
- Fire Emergency

## Project structure

- `app.py` – Streamlit dashboard and ML workflow
- `data/incident_data.csv` – dataset used by the app
- `requirements.txt` – Python dependencies
