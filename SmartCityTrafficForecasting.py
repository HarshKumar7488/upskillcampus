
import io
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Smart City Traffic Forecasting",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
        background: #f5f7fb !important;
        color: #111827 !important;
    }

    [data-testid="stHeader"] {
        background: #f5f7fb !important;
    }

    [data-testid="stSidebar"] {
        background: #ffffff !important;
        border-right: 1px solid #e5e7eb;
    }

    [data-testid="stSidebar"] * {
        color: #111827 !important;
    }

    .main {
        background-color: #f5f7fb !important;
    }

    .block-container {
        padding-top: 1.5rem;
    }

    .hero {
        background: linear-gradient(135deg, #102a56, #2563eb);
        padding: 28px;
        border-radius: 18px;
        color: white !important;
        margin-bottom: 20px;
        box-shadow: 0 8px 25px rgba(15,23,42,.12);
    }

    .hero h1, .hero p {
        color: white !important;
    }

    .hero h1 {
        margin: 0;
        font-size: 2.1rem;
    }

    .hero p {
        margin-top: 8px;
        opacity: .9;
    }

    .info-box {
        padding: 14px;
        border-radius: 12px;
        background: #ffffff !important;
        color: #111827 !important;
        border-left: 4px solid #2563eb;
        margin-bottom: 10px;
        box-shadow: 0 2px 10px rgba(15,23,42,.08);
    }

    .success-box {
        padding: 14px;
        border-radius: 12px;
        background: #ecfdf5 !important;
        color: #14532d !important;
        border-left: 4px solid #16a34a;
    }

    .warning-box {
        padding: 14px;
        border-radius: 12px;
        background: #fffbeb !important;
        color: #713f12 !important;
        border-left: 4px solid #f59e0b;
    }

    .danger-box {
        padding: 14px;
        border-radius: 12px;
        background: #fef2f2 !important;
        color: #7f1d1d !important;
        border-left: 4px solid #dc2626;
    }

    [data-testid="stMetric"] {
        background: #ffffff !important;
        color: #111827 !important;
        padding: 14px;
        border-radius: 14px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 5px 18px rgba(15,23,42,.10);
    }

    [data-testid="stMetric"] label,
    [data-testid="stMetric"] [data-testid="stMetricLabel"],
    [data-testid="stMetric"] [data-testid="stMetricValue"],
    [data-testid="stMetric"] [data-testid="stMetricDelta"] {
        color: #111827 !important;
    }

    [data-testid="stDataFrame"], [data-testid="stTable"] {
        background: #ffffff !important;
    }

    [data-testid="stFileUploader"] {
        background: #ffffff !important;
        border-radius: 12px;
    }

    [data-baseweb="select"] > div,
    [data-testid="stSelectbox"] > div > div {
        background: #ffffff !important;
        color: #111827 !important;
    }

    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stMarkdownContainer"] span,
    h1, h2, h3, h4, h5, h6, label {
        color: #111827;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

JUNCTION_CONFIG = {
    "MP Nagar Junction": {"base": 820, "speed": 42, "capacity": 1050},
    "Habibganj Square": {"base": 690, "speed": 46, "capacity": 950},
    "Lalghati Junction": {"base": 610, "speed": 48, "capacity": 900},
    "Board Office Square": {"base": 760, "speed": 43, "capacity": 1000},
    "Ayodhya Bypass": {"base": 930, "speed": 40, "capacity": 1150},
}

RANDOM_STATE = 42

@st.cache_data
def generate_dataset(days: int = 120) -> pd.DataFrame:

    rng = np.random.default_rng(RANDOM_STATE)

    start = (
        pd.Timestamp.now()
        .floor("h")
        - pd.Timedelta(days=days)
    )

    timestamps = pd.date_range(
        start=start,
        periods=days * 24,
        freq="h",
    )

    records = []

    for junction, config in JUNCTION_CONFIG.items():

        for timestamp in timestamps:

            hour = timestamp.hour
            day_of_week = timestamp.dayofweek
            month = timestamp.month

            weekend = int(day_of_week >= 5)

            morning_peak = np.exp(
                -((hour - 9) / 2.3) ** 2
            )

            evening_peak = np.exp(
                -((hour - 18) / 2.8) ** 2
            )

            midday = np.exp(
                -((hour - 13) / 4.5) ** 2
            )

            night_factor = -0.25 if hour < 5 else 0

            seasonal_factor = (
                1
                + 0.04 * np.sin(
                    2 * np.pi * timestamp.dayofyear / 365
                )
            )

            weekend_factor = (
                0.72 if weekend else 1.0
            )

            event = int(
                rng.random() < 0.025
            )

            rain = max(
                0,
                rng.normal(0.15, 0.25)
            )

            temperature = (
                25
                + 7 * np.sin(
                    2 * np.pi * timestamp.dayofyear / 365
                )
                + rng.normal(0, 2)
            )

            demand_multiplier = (
                1
                + 0.78 * morning_peak
                + 0.95 * evening_peak
                + 0.18 * midday
                + night_factor
            )

            demand_multiplier *= weekend_factor
            demand_multiplier *= seasonal_factor

            if event:
                demand_multiplier *= 1.18

            if rain > 0.45:
                demand_multiplier *= 1.08

            noise = rng.normal(
                0,
                config["base"] * 0.045,
            )

            vehicles = max(
                50,
                int(
                    config["base"]
                    * demand_multiplier
                    + noise
                ),
            )

            speed = (
                config["speed"]
                - 18
                * (
                    vehicles
                    / config["capacity"]
                )
                + rng.normal(0, 2)
            )

            speed = float(
                np.clip(
                    speed,
                    12,
                    65,
                )
            )

            records.append(
                {
                    "timestamp": timestamp,
                    "junction": junction,
                    "vehicles": vehicles,
                    "speed": round(speed, 2),
                    "temperature": round(
                        temperature,
                        2,
                    ),
                    "rain": round(
                        rain,
                        3,
                    ),
                    "event": event,
                    "hour": hour,
                    "day_of_week": day_of_week,
                    "month": month,
                    "is_weekend": weekend,
                }
            )

    return pd.DataFrame(records)

def prepare_uploaded_data(uploaded_file) -> pd.DataFrame:

    df = pd.read_csv(uploaded_file)

    original_columns = list(df.columns)

    normalized = {
        str(c).strip().lower().replace(" ", "_"): c
        for c in df.columns
    }

    timestamp_column = None

    for candidate in [
        "timestamp",
        "datetime",
        "date_time",
        "date",
        "time",
    ]:
        if candidate in normalized:
            timestamp_column = normalized[candidate]
            break

    if timestamp_column is None:
        raise ValueError(
            "CSV must contain a timestamp/datetime/date/time column."
        )

    junction_column = None

    for candidate in [
        "junction",
        "location",
        "intersection",
        "site",
    ]:
        if candidate in normalized:
            junction_column = normalized[candidate]
            break

    traffic_column = None

    for candidate in [
        "vehicles",
        "traffic",
        "volume",
        "vehicle_count",
        "traffic_volume",
    ]:
        if candidate in normalized:
            traffic_column = normalized[candidate]
            break

    if traffic_column is None:
        raise ValueError(
            "CSV must contain a traffic column such as vehicles, "
            "traffic, volume or vehicle_count."
        )

    result = pd.DataFrame()

    result["timestamp"] = pd.to_datetime(
        df[timestamp_column],
        errors="coerce",
    )

    if junction_column is not None:
        result["junction"] = (
            df[junction_column]
            .astype(str)
            .str.strip()
        )
    else:
        result["junction"] = "Uploaded Junction"

    result["vehicles"] = pd.to_numeric(
        df[traffic_column],
        errors="coerce",
    )

    result = result.dropna(
        subset=["timestamp", "vehicles"]
    )

    result = result.sort_values(
        "timestamp"
    )

    result["hour"] = result["timestamp"].dt.hour
    result["day_of_week"] = (
        result["timestamp"].dt.dayofweek
    )
    result["month"] = result["timestamp"].dt.month
    result["is_weekend"] = (
        result["day_of_week"] >= 5
    ).astype(int)

    if "speed" in normalized:
        result["speed"] = pd.to_numeric(
            df.loc[
                result.index,
                normalized["speed"],
            ],
            errors="coerce",
        )
    else:
        result["speed"] = np.nan

    if "rain" in normalized:
        result["rain"] = pd.to_numeric(
            df.loc[
                result.index,
                normalized["rain"],
            ],
            errors="coerce",
        )
    else:
        result["rain"] = 0.0

    if "temperature" in normalized:
        result["temperature"] = pd.to_numeric(
            df.loc[
                result.index,
                normalized["temperature"],
            ],
            errors="coerce",
        )
    else:
        result["temperature"] = 25.0

    if "event" in normalized:
        result["event"] = pd.to_numeric(
            df.loc[
                result.index,
                normalized["event"],
            ],
            errors="coerce",
        ).fillna(0)
    else:
        result["event"] = 0

    result = result.reset_index(drop=True)

    return result

FEATURE_COLUMNS = [
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_24",
    "rolling_3",
    "rolling_6",
    "rolling_24",
    "temperature",
    "rain",
    "event",
]

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:

    result = df.copy()

    result = result.sort_values(
        ["junction", "timestamp"]
    )

    grouped = result.groupby(
        "junction",
        group_keys=False,
    )

    result["lag_1"] = grouped["vehicles"].shift(1)
    result["lag_2"] = grouped["vehicles"].shift(2)
    result["lag_3"] = grouped["vehicles"].shift(3)
    result["lag_24"] = grouped["vehicles"].shift(24)

    result["rolling_3"] = (
        grouped["vehicles"]
        .shift(1)
        .rolling(3)
        .mean()
        .reset_index(level=0, drop=True)
    )

    result["rolling_6"] = (
        grouped["vehicles"]
        .shift(1)
        .rolling(6)
        .mean()
        .reset_index(level=0, drop=True)
    )

    result["rolling_24"] = (
        grouped["vehicles"]
        .shift(1)
        .rolling(24)
        .mean()
        .reset_index(level=0, drop=True)
    )

    return result

@st.cache_resource
def train_models(df: pd.DataFrame):

    featured = engineer_features(df)

    featured = featured.dropna(
        subset=FEATURE_COLUMNS + ["vehicles"]
    ).copy()

    if len(featured) < 100:
        raise ValueError(
            "Not enough clean observations for model training."
        )

    split_index = int(
        len(featured) * 0.80
    )

    train = featured.iloc[:split_index]
    test = featured.iloc[split_index:]

    X_train = train[FEATURE_COLUMNS]
    y_train = train["vehicles"]

    X_test = test[FEATURE_COLUMNS]
    y_test = test["vehicles"]

    rf = RandomForestRegressor(
        n_estimators=250,
        max_depth=18,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    gb = GradientBoostingRegressor(
        n_estimators=220,
        learning_rate=0.05,
        max_depth=4,
        loss="huber",
        random_state=RANDOM_STATE,
    )

    rf.fit(
        X_train,
        y_train,
    )

    gb.fit(
        X_train,
        y_train,
    )

    rf_prediction = rf.predict(X_test)
    gb_prediction = gb.predict(X_test)

    rf_mae = mean_absolute_error(
        y_test,
        rf_prediction,
    )

    gb_mae = mean_absolute_error(
        y_test,
        gb_prediction,
    )

    if rf_mae <= gb_mae:
        model = rf
        model_name = "Random Forest Regressor"
        prediction = rf_prediction
    else:
        model = gb
        model_name = "Gradient Boosting Regressor"
        prediction = gb_prediction

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            prediction,
        )
    )

    r2 = r2_score(
        y_test,
        prediction,
    )

    mae = mean_absolute_error(
        y_test,
        prediction,
    )

    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Random Forest MAE": rf_mae,
        "Gradient Boosting MAE": gb_mae,
        "model_name": model_name,
    }

    importance = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance": model.feature_importances_,
        }
    ).sort_values(
        "importance",
        ascending=False,
    )

    return (
        model,
        metrics,
        importance,
        featured,
    )

def make_forecast(
    model,
    history: pd.DataFrame,
    junction: str,
    horizon: int,
):

    data = history.copy()

    data = data[
        data["junction"] == junction
    ].sort_values(
        "timestamp"
    ).copy()

    if len(data) < 30:
        raise ValueError(
            "At least 30 observations are required "
            "for forecasting."
        )

    values = data["vehicles"].astype(float).tolist()

    last_timestamp = data["timestamp"].iloc[-1]

    last_temperature = (
        data["temperature"].iloc[-1]
        if "temperature" in data
        else 25
    )

    last_rain = (
        data["rain"].iloc[-1]
        if "rain" in data
        else 0
    )

    last_event = (
        data["event"].iloc[-1]
        if "event" in data
        else 0
    )

    forecasts = []

    for step in range(1, horizon + 1):

        timestamp = (
            last_timestamp
            + pd.Timedelta(hours=step)
        )

        hour = timestamp.hour
        day_of_week = timestamp.dayofweek
        month = timestamp.month
        is_weekend = int(
            day_of_week >= 5
        )

        lag_1 = values[-1]
        lag_2 = values[-2]
        lag_3 = values[-3]
        lag_24 = (
            values[-24]
            if len(values) >= 24
            else np.mean(values)
        )

        rolling_3 = np.mean(
            values[-3:]
        )

        rolling_6 = np.mean(
            values[-6:]
        )

        rolling_24 = np.mean(
            values[-24:]
        )

        row = pd.DataFrame(
            [
                {
                    "hour": hour,
                    "day_of_week": day_of_week,
                    "month": month,
                    "is_weekend": is_weekend,
                    "lag_1": lag_1,
                    "lag_2": lag_2,
                    "lag_3": lag_3,
                    "lag_24": lag_24,
                    "rolling_3": rolling_3,
                    "rolling_6": rolling_6,
                    "rolling_24": rolling_24,
                    "temperature": last_temperature,
                    "rain": last_rain,
                    "event": last_event,
                }
            ]
        )

        prediction = float(
            model.predict(row[FEATURE_COLUMNS])[0]
        )

        prediction = max(
            0,
            prediction,
        )

        values.append(prediction)

        forecasts.append(
            {
                "timestamp": timestamp,
                "junction": junction,
                "predicted_vehicles": round(
                    prediction
                ),
                "hour": hour,
            }
        )

    result = pd.DataFrame(
        forecasts
    )

    if junction in JUNCTION_CONFIG:

        base_speed = JUNCTION_CONFIG[
            junction
        ]["speed"]

        capacity = JUNCTION_CONFIG[
            junction
        ]["capacity"]

        result["predicted_speed"] = (
            base_speed
            -
            18
            *
            (
                result[
                    "predicted_vehicles"
                ]
                /
                capacity
            )
        ).clip(
            lower=12,
            upper=65,
        ).round(1)

    else:

        result["predicted_speed"] = np.nan

    return result

def congestion_label(
    vehicles,
    junction,
):

    if junction in JUNCTION_CONFIG:

        capacity = JUNCTION_CONFIG[
            junction
        ]["capacity"]

        ratio = vehicles / capacity

    else:

        ratio = 0.75

    if ratio >= 0.90:
        return "HIGH"

    if ratio >= 0.65:
        return "MEDIUM"

    return "LOW"

def detect_anomalies(
    df: pd.DataFrame,
) -> pd.DataFrame:

    result = df.copy()

    q1 = result.groupby(
        "junction"
    )["vehicles"].transform(
        lambda x: x.quantile(0.25)
    )

    q3 = result.groupby(
        "junction"
    )["vehicles"].transform(
        lambda x: x.quantile(0.75)
    )

    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    result["anomaly"] = (
        (result["vehicles"] < lower)
        |
        (result["vehicles"] > upper)
    )

    return result

def traffic_chart(
    historical,
    forecast,
):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=historical["timestamp"],
            y=historical["vehicles"],
            mode="lines",
            name="Historical Traffic",
            line=dict(
                color="#2563eb",
                width=3,
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=forecast["timestamp"],
            y=forecast["predicted_vehicles"],
            mode="lines+markers",
            name="Forecast",
            line=dict(
                color="#f59e0b",
                width=3,
                dash="dash",
            ),
        )
    )

    if not forecast.empty:

        peak_index = forecast[
            "predicted_vehicles"
        ].idxmax()

        peak_row = forecast.loc[
            peak_index
        ]

        fig.add_trace(
            go.Scatter(
                x=[
                    peak_row["timestamp"]
                ],
                y=[
                    peak_row[
                        "predicted_vehicles"
                    ]
                ],
                mode="markers+text",
                name="Forecast Peak",
                text=["Peak"],
                textposition="top center",
                marker=dict(
                    size=12,
                    color="#dc2626",
                ),
            )
        )

    fig.update_layout(
        height=420,
        template="plotly_white",
        hovermode="x unified",
        xaxis_title="Time",
        yaxis_title="Vehicles / Hour",
        legend_title="Traffic",
        margin=dict(
            l=20,
            r=20,
            t=20,
            b=20,
        ),
    )

    return fig

def speed_chart(forecast):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=forecast["timestamp"],
            y=forecast["predicted_speed"],
            mode="lines+markers",
            name="Predicted Speed",
            line=dict(
                color="#16a34a",
                width=3,
            ),
        )
    )

    fig.add_hline(
        y=30,
        line_dash="dash",
        line_color="#dc2626",
        annotation_text="Slow traffic threshold",
    )

    fig.update_layout(
        height=330,
        template="plotly_white",
        xaxis_title="Time",
        yaxis_title="Speed (km/h)",
    )

    return fig

def create_report(
    junction,
    forecast,
    metrics,
    generated_at,
):
    lines = [
        "SMART CITY TRAFFIC FORECASTING REPORT",
        "=" * 50,
        "",
        f"Junction: {junction}",
        f"Generated: {generated_at}",
        "",
        "MODEL",
        "-" * 50,
        f"Model: {metrics['model_name']}",
        f"MAE: {metrics['MAE']:.2f}",
        f"RMSE: {metrics['RMSE']:.2f}",
        f"R2: {metrics['R2']:.4f}",
        "",
        "FORECAST",
        "-" * 50,
    ]

    for _, row in forecast.iterrows():

        lines.append(
            f"{row['timestamp']} | "
            f"{row['predicted_vehicles']} vehicles/hour | "
            f"{row['predicted_speed']:.1f} km/h"
        )

    peak = forecast.loc[
        forecast["predicted_vehicles"].idxmax()
    ]

    lines.extend(
        [
            "",
            "SUMMARY",
            "-" * 50,
            f"Peak traffic: "
            f"{peak['predicted_vehicles']} vehicles/hour",
            f"Peak time: {peak['timestamp']}",
            "",
            "This report was generated from the selected traffic data.",
        ]
    )

    return "\n".join(lines)

st.markdown(
    """
    <div class="hero">
        <h1>🚦 Smart City Traffic Forecasting</h1>
        <p>
        Traffic volume forecasting, congestion monitoring,
        anomaly detection and junction-level analysis.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:

    st.header("⚙️ Controls")

    uploaded_file = st.file_uploader(
        "Upload traffic data (CSV)",
        type=["csv"],
        help=(
            "CSV should contain timestamp, junction and "
            "vehicles/traffic/volume columns."
        ),
    )

    st.divider()

    forecast_horizon = st.selectbox(
        "Forecast Horizon",
        [6, 12, 24, 48],
        index=1,
        format_func=lambda x: f"Next {x} Hours",
    )

    use_demo = st.checkbox(
        "Use sample dataset",
        value=uploaded_file is None,
    )

    st.divider()

    st.subheader("Tools Used")

    st.write(
        """
      
        """
    )

try:

    if uploaded_file is not None and not use_demo:

        data = prepare_uploaded_data(
            uploaded_file
        )

        data_source = "Uploaded CSV"

    else:

        data = generate_dataset(
            days=120
        )

        data_source = (
            "Sample traffic dataset"
        )

except Exception as exc:

    st.error(
        f"Unable to load data: {exc}"
    )

    st.stop()

available_junctions = sorted(
    data["junction"]
    .dropna()
    .unique()
    .tolist()
)

if not available_junctions:

    st.error(
        "No junctions were found in the dataset."
    )

    st.stop()

with st.sidebar:

    selected_junction = st.selectbox(
        "Select Junction",
        available_junctions,
    )

with st.spinner(
    "Preparing the forecasting models..."
):

    try:

        (
            model,
            metrics,
            feature_importance,
            featured_data,
        ) = train_models(
            data
        )

    except Exception as exc:

        st.error(
            f"Model training failed: {exc}"
        )

        st.stop()

try:

    forecast = make_forecast(
        model=model,
        history=data,
        junction=selected_junction,
        horizon=forecast_horizon,
    )

except Exception as exc:

    st.error(
        f"Forecast generation failed: {exc}"
    )

    st.stop()

junction_history = data[
    data["junction"]
    == selected_junction
].sort_values(
    "timestamp"
).copy()

recent_history = junction_history.tail(
    48
)

current_traffic = float(
    junction_history[
        "vehicles"
    ].iloc[-1]
)

previous_traffic = (
    float(
        junction_history[
            "vehicles"
        ].iloc[-2]
    )
    if len(junction_history) >= 2
    else current_traffic
)

peak_row = forecast.loc[
    forecast[
        "predicted_vehicles"
    ].idxmax()
]

peak_traffic = float(
    peak_row[
        "predicted_vehicles"
    ]
)

average_forecast = float(
    forecast[
        "predicted_vehicles"
    ].mean()
)

average_speed = float(
    forecast[
        "predicted_speed"
    ].mean()
)

current_status = congestion_label(
    current_traffic,
    selected_junction,
)

peak_status = congestion_label(
    peak_traffic,
    selected_junction,
)

st.subheader(
    f"📊 Live Traffic Dashboard — {selected_junction}"
)

col1, col2, col3, col4 = st.columns(4)

with col1:

    delta = (
        current_traffic
        -
        previous_traffic
    )

    st.metric(
        "Current Vehicles / Hour",
        f"{current_traffic:,.0f}",
        f"{delta:+,.0f}",
    )

with col2:

    st.metric(
        "Forecast Peak",
        f"{peak_traffic:,.0f}",
        peak_row["timestamp"].strftime(
            "%d %b %H:%M"
        ),
    )

with col3:

    st.metric(
        "Average Forecast Speed",
        f"{average_speed:.1f} km/h",
        "Forecast period",
    )

with col4:

    st.metric(
        "Current Congestion",
        current_status,
        f"Peak: {peak_status}",
    )

left, right = st.columns(
    [2, 1]
)

with left:

    st.subheader(
        "📈 Traffic History and Forecast"
    )

    st.plotly_chart(
        traffic_chart(
            recent_history,
            forecast,
        ),
        use_container_width=True,
    )

with right:

    st.subheader(
        "📋 Forecast Details"
    )

    st.markdown(
        f"""
        <div class="info-box">

        <b>Forecasting model:</b><br>
        {metrics["model_name"]}

        </div>

        <div class="info-box">

        <b>Next hour:</b><br>
        {forecast.iloc[0]["predicted_vehicles"]:,}
        vehicles/hour

        </div>

        <div class="warning-box">

        <b>Peak traffic:</b><br>
        {peak_traffic:,.0f}
        vehicles/hour<br>
        {peak_row["timestamp"].strftime("%d %b %Y, %H:%M")}

        </div>
        """,
        unsafe_allow_html=True,
    )

    if average_speed < 30:

        st.markdown(
            """
            <div class="danger-box">
            ⚠️ Average forecast speed is below
            30 km/h, indicating heavier traffic.
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:

        st.markdown(
            """
            <div class="success-box">
            ✓ Average forecast speed indicates
            relatively steady traffic flow.
            </div>
            """,
            unsafe_allow_html=True,
        )

left, right = st.columns(2)

with left:

    st.subheader(
        "🚗 Forecast Speed"
    )

    st.plotly_chart(
        speed_chart(forecast),
        use_container_width=True,
    )

with right:

    st.subheader(
        "🧠 Important Predictors"
    )

    top_features = (
        feature_importance
        .head(10)
        .sort_values(
            "importance"
        )
    )

    fig = go.Figure(
        go.Bar(
            x=top_features[
                "importance"
            ],
            y=top_features[
                "feature"
            ],
            orientation="h",
            marker_color="#2563eb",
        )
    )

    fig.update_layout(
        height=330,
        template="plotly_white",
        xaxis_title="Importance",
        yaxis_title="Feature",
        margin=dict(
            l=20,
            r=20,
            t=20,
            b=20,
        ),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

st.subheader(
    "🎯 Model Accuracy"
)

m1, m2, m3, m4 = st.columns(4)

with m1:

    st.metric(
        "MAE",
        f"{metrics['MAE']:.2f}",
    )

with m2:

    st.metric(
        "RMSE",
        f"{metrics['RMSE']:.2f}",
    )

with m3:

    st.metric(
        "R² Score",
        f"{metrics['R2']:.4f}",
    )

with m4:

    st.metric(
        "Model",
        metrics["model_name"],
    )

st.caption(
    "Model results are measured on the last 20% of the time-ordered data."
)

st.subheader(
    "🕐 Hourly Forecast"
)

display_forecast = forecast.copy()

display_forecast[
    "congestion"
] = display_forecast.apply(
    lambda row: congestion_label(
        row["predicted_vehicles"],
        selected_junction,
    ),
    axis=1,
)

display_forecast[
    "timestamp"
] = display_forecast[
    "timestamp"
].dt.strftime(
    "%d %b %Y %H:%M"
)

display_forecast = display_forecast.rename(
    columns={
        "timestamp": "Time",
        "predicted_vehicles": "Predicted Vehicles/Hour",
        "predicted_speed": "Predicted Speed (km/h)",
        "congestion": "Congestion",
    }
)

st.dataframe(
    display_forecast[
        [
            "Time",
            "Predicted Vehicles/Hour",
            "Predicted Speed (km/h)",
            "Congestion",
        ]
    ],
    use_container_width=True,
    hide_index=True,
)

st.subheader(
    "🏙️ Junctions"
)

comparison_rows = []

for junction in available_junctions:

    junction_df = data[
        data["junction"]
        == junction
    ].sort_values(
        "timestamp"
    )

    try:

        junction_forecast = make_forecast(
            model=model,
            history=data,
            junction=junction,
            horizon=forecast_horizon,
        )

        current = float(
            junction_df[
                "vehicles"
            ].iloc[-1]
        )

        predicted_peak = float(
            junction_forecast[
                "predicted_vehicles"
            ].max()
        )

        speed = float(
            junction_forecast[
                "predicted_speed"
            ].mean()
        )

        comparison_rows.append(
            {
                "Junction": junction,
                "Current Vehicles/Hour": round(
                    current
                ),
                "Predicted Peak": round(
                    predicted_peak
                ),
                "Avg Forecast Speed": round(
                    speed,
                    1,
                ),
                "Congestion": congestion_label(
                    current,
                    junction,
                ),
            }
        )

    except Exception:

        continue

comparison_df = pd.DataFrame(
    comparison_rows
)

st.dataframe(
    comparison_df,
    use_container_width=True,
    hide_index=True,
)

st.subheader(
    "🔍 Unusual Traffic Records"
)

anomaly_data = detect_anomalies(
    junction_history
)

anomalies = anomaly_data[
    anomaly_data["anomaly"]
].tail(20)

if anomalies.empty:

    st.success(
        "No significant traffic-volume anomalies "
        "were detected using the IQR method."
    )

else:

    st.warning(
        f"{len(anomalies)} recent anomaly/anomalies "
        "were detected."
    )

    anomaly_display = anomalies[
        [
            "timestamp",
            "vehicles",
        ]
    ].copy()

    anomaly_display[
        "timestamp"
    ] = anomaly_display[
        "timestamp"
    ].dt.strftime(
        "%d %b %Y %H:%M"
    )

    st.dataframe(
        anomaly_display,
        use_container_width=True,
        hide_index=True,
    )

st.subheader(
    "🧠 Traffic Summary"
)

peak_hour = peak_row[
    "timestamp"
].hour

morning_peak = (
    peak_hour >= 7
    and peak_hour <= 11
)

evening_peak = (
    peak_hour >= 16
    and peak_hour <= 21
)

if morning_peak:

    peak_description = (
        "The forecast peak occurs during the morning "
        "commuter period."
    )

elif evening_peak:

    peak_description = (
        "The forecast peak occurs during the evening "
        "commuter period."
    )

else:

    peak_description = (
        "The forecast peak occurs outside the major "
        "commuter periods."
    )

intelligence = [
    (
        "Peak Period",
        peak_description,
    ),
    (
        "Expected Demand",
        f"Average predicted demand is "
        f"{average_forecast:,.0f} vehicles/hour "
        f"over the selected horizon.",
    ),
    (
        "Congestion",
        f"Current traffic level is {current_status}. "
        f"The forecast peak is {peak_status}.",
    ),
    (
        "Planning Note",
        (
            "Traffic staff can review signal timing and "
            "diversion options before the predicted peak "
            "period."
        ),
    ),
]

for title, description in intelligence:

    st.markdown(
        f"""
        <div class="info-box">
        <b>{title}</b><br>
        {description}
        </div>
        """,
        unsafe_allow_html=True,
    )

with st.expander(
    "📋 View Data"
):

    st.dataframe(
        data.tail(100),
        use_container_width=True,
        hide_index=True,
    )

st.subheader(
    "📥 Export Results"
)

generated_at = datetime.now().strftime(
    "%d %B %Y, %H:%M:%S"
)

report_text = create_report(
    selected_junction,
    forecast,
    metrics,
    generated_at,
)

csv_bytes = forecast.to_csv(
    index=False
).encode(
    "utf-8"
)

excel_buffer = io.BytesIO()

with pd.ExcelWriter(
    excel_buffer,
    engine="openpyxl",
) as writer:

    forecast.to_excel(
        writer,
        sheet_name="Forecast",
        index=False,
    )

    comparison_df.to_excel(
        writer,
        sheet_name="Junctions",
        index=False,
    )

    feature_importance.to_excel(
        writer,
        sheet_name="Feature Importance",
        index=False,
    )

download_col1, download_col2, download_col3 = st.columns(3)

with download_col1:

    st.download_button(
        label="📄 Download Text Report",
        data=report_text,
        file_name="traffic_forecast_report.txt",
        mime="text/plain",
    )

with download_col2:

    st.download_button(
        label="📊 Download Forecast CSV",
        data=csv_bytes,
        file_name="traffic_forecast.csv",
        mime="text/csv",
    )

with download_col3:

    st.download_button(
        label="📗 Download Excel Report",
        data=excel_buffer.getvalue(),
        file_name="traffic_forecast_report.xlsx",
        mime=(
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        ),
    )


