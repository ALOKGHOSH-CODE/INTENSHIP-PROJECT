from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path("data/incident_data.csv")
TARGET_COLUMN = "incident_type"
FEATURE_COLUMNS = [
    "temperature_c",
    "smoke_level",
    "co_ppm",
    "gas_level",
    "sound_db",
    "crowd_count",
    "motion_index",
    "flame_detected",
    "event_duration_min",
]

st.set_page_config(page_title="Industrial Safety Dashboard", layout="wide")


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["severity_label"] = df["severity_label"].astype(str)
    return df


@st.cache_data
def train_incident_model(df: pd.DataFrame):
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=250,
                    max_depth=None,
                    min_samples_leaf=1,
                    random_state=42,
                    class_weight="balanced_subsample",
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(
        y_test,
        predictions,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_test, predictions, labels=model.classes_)

    feature_importance = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance": model.named_steps["classifier"].feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    return model, accuracy, report, cm, feature_importance


@st.cache_data
def get_summary_metrics(df: pd.DataFrame):
    safe_count = int((df[TARGET_COLUMN] == "Safe / Normal Activity").sum())
    alert_count = int((df["severity"] >= 1).sum())
    critical_count = int((df["severity"] >= 2).sum())
    avg_temp = float(df["temperature_c"].mean())
    avg_gas = float(df["gas_level"].mean())
    return {
        "records": int(len(df)),
        "safe_count": safe_count,
        "alert_count": alert_count,
        "critical_count": critical_count,
        "avg_temp": avg_temp,
        "avg_gas": avg_gas,
    }


def render_overview(df: pd.DataFrame):
    metrics = get_summary_metrics(df)

    st.title("Industrial Safety Monitoring Dashboard")
    st.caption("AI-powered analysis of facility conditions and emergency patterns")

    cols = st.columns(5)
    cols[0].metric("Total records", f"{metrics['records']}")
    cols[1].metric("Safe activities", f"{metrics['safe_count']}")
    cols[2].metric("Alerts logged", f"{metrics['alert_count']}")
    cols[3].metric("Critical incidents", f"{metrics['critical_count']}")
    cols[4].metric("Avg. temp", f"{metrics['avg_temp']:.1f}°C")

    incident_counts = df[TARGET_COLUMN].value_counts().sort_values(ascending=False)
    incident_counts_df = incident_counts.reset_index()
    incident_counts_df.columns = ["Incident Type", "Count"]
    fig_counts = px.bar(
        incident_counts_df,
        x="Incident Type",
        y="Count",
        color="Incident Type",
        title="Incident Frequency by Type",
    )
    fig_counts.update_layout(showlegend=False, xaxis_tickangle=-35)

    severity_counts = df["severity_label"].value_counts().sort_index()
    severity_counts_df = severity_counts.reset_index()
    severity_counts_df.columns = ["Severity", "Count"]
    fig_severity = px.pie(
        severity_counts_df,
        names="Severity",
        values="Count",
        title="Severity Distribution",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )

    left_col, right_col = st.columns(2)
    left_col.plotly_chart(fig_counts, use_container_width=True)
    right_col.plotly_chart(fig_severity, use_container_width=True)

    corr = df[FEATURE_COLUMNS].corr(numeric_only=True)
    fig_corr = px.imshow(
        corr,
        title="Sensor Correlation Matrix",
        color_continuous_scale="RdBu_r",
        aspect="auto",
    )
    st.plotly_chart(fig_corr, use_container_width=True)


def render_model_section(df: pd.DataFrame):
    model, accuracy, report, cm, feature_importance = train_incident_model(df)

    st.subheader("Incident Prediction Model")
    st.write(
        "This predictive model checks the key environment indicators and estimates the most likely incident type."
    )

    st.markdown(f"### Model accuracy: {accuracy:.2%}")

    report_df = pd.DataFrame(report).T
    report_df = report_df.reset_index().rename(columns={"index": "class"})
    st.dataframe(report_df, use_container_width=True)

    confusion_df = pd.DataFrame(
        cm,
        index=model.classes_,
        columns=model.classes_,
    )
    fig_cm = px.imshow(
        confusion_df,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Blues",
        title="Confusion Matrix",
    )
    st.plotly_chart(fig_cm, use_container_width=True)

    st.subheader("Top Risk Factors")
    fig_importance = px.bar(
        feature_importance,
        x="importance",
        y="feature",
        orientation="h",
        color="importance",
        title="Feature Importance",
        color_continuous_scale="Viridis",
    )
    st.plotly_chart(fig_importance, use_container_width=True)


def render_prediction_panel(model):
    st.subheader("Live Incident Prediction")
    st.write("Adjust the sensor values below to estimate the current incident category.")

    with st.form("prediction_form"):
        input_values = {}
        input_values["temperature_c"] = st.slider("Temperature (°C)", 10.0, 70.0, 30.0, 0.1)
        input_values["smoke_level"] = st.slider("Smoke level", 0.0, 100.0, 20.0, 0.1)
        input_values["co_ppm"] = st.slider("CO (ppm)", 0.0, 120.0, 15.0, 0.1)
        input_values["gas_level"] = st.slider("Gas level", 0.0, 100.0, 25.0, 0.1)
        input_values["sound_db"] = st.slider("Sound (dB)", 20.0, 150.0, 60.0, 1.0)
        input_values["crowd_count"] = st.slider("Crowd count", 0, 100, 20, 1)
        input_values["motion_index"] = st.slider("Motion index", 0.0, 100.0, 30.0, 0.1)
        input_values["flame_detected"] = st.selectbox("Flame detected", [0, 1])
        input_values["event_duration_min"] = st.slider("Event duration (min)", 0.0, 15.0, 3.0, 0.1)

        submitted = st.form_submit_button("Predict incident")

    if submitted:
        prediction_df = pd.DataFrame([input_values], columns=FEATURE_COLUMNS)
        prediction = model.predict(prediction_df)[0]
        probabilities = model.predict_proba(prediction_df)[0]
        confidence = float(np.max(probabilities))
        probability_df = pd.DataFrame(
            {
                "Incident Type": model.classes_,
                "Probability": probabilities,
            }
        ).sort_values("Probability", ascending=False)

        st.success(f"Predicted incident: {prediction}")
        st.metric("Confidence", f"{confidence * 100:.1f}%")
        st.dataframe(probability_df, use_container_width=True)


def main():
    df = load_data()

    tabs = st.tabs(["Overview", "Model", "Prediction"])

    with tabs[0]:
        render_overview(df)

    with tabs[1]:
        _, _, _, _, _ = train_incident_model(df)
        render_model_section(df)

    with tabs[2]:
        model, _, _, _, _ = train_incident_model(df)
        render_prediction_panel(model)


if __name__ == "__main__":
    main()
