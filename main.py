from keras.models import load_model
import datetime
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
from datetime import timedelta, timezone
from sklearn.preprocessing import MinMaxScaler
from pandas_market_calendars.calendars.iex import DatetimeIndex
import gradio as gr


def add_nyse_days(start_date: datetime.datetime, num_days: int) -> datetime.datetime:
    nyse = mcal.get_calendar("NYSE")
    schedule: pd.DataFrame = nyse.schedule(
        start_date=start_date, end_date=start_date + timedelta(days=365)
    )
    return schedule.index[num_days].to_pydatetime()


def remove_nyse_days(start_date: datetime.datetime, num_days: int) -> datetime.datetime:
    nyse = mcal.get_calendar("NYSE")
    schedule: pd.DataFrame = nyse.schedule(
        start_date=(start_date - timedelta(days=365)), end_date=start_date
    )
    return schedule.index[num_days].to_pydatetime()


def nyse_days(start_date: datetime.datetime) -> pd.Index:
    nyse = mcal.get_calendar("NYSE")
    schedule: pd.DataFrame = nyse.schedule(
        start_date=start_date, end_date=start_date + timedelta(days=365)
    )
    return schedule.index


def predict_model(start_date: datetime.datetime, firm: str) -> pd.DataFrame:
    firm_data: pd.DataFrame = np.load(f"data/{firm}.npz", allow_pickle=True)
    # print(firm_data.head())
    model = load_model(f"models/SuperLong.h5", safe_mode=False)
    target_ts = start_date.replace(tzinfo=timezone.utc).timestamp()
    date_index = (firm_data["UnixTime"] - target_ts).abs().idxmin()

    scaler = MinMaxScaler(feature_range=(0, 1))
    date_df: pd.DataFrame = firm_data.iloc[date_index : date_index + 60]
    date_df.loc[:, "Close"] = scaler.fit_transform(date_df[["Close"]])
    model_prediction_input = date_df.drop("UnixTime", axis=1).to_numpy()
    model_prediction_input = np.array([model_prediction_input])
    output_df = pd.DataFrame(model.predict(model_prediction_input)[0])
    output_df.columns = ["Close", "Predicted Sentiment"]
    output_df = output_df.drop("Predicted Sentiment", axis=1)
    output_df.loc[:, "Close"] = scaler.inverse_transform(output_df[["Close"]])
    current_date = datetime.datetime.fromtimestamp(date_df.iloc[59]["UnixTime"])
    output_df.loc[0, "UnixTime"] = current_date.timestamp()
    for i in range(1, 30):
        current_date = add_nyse_days(current_date, 1)
        output_df.loc[i, "UnixTime"] = current_date.timestamp()
    before_data = (np.load(f"data/{firm}.npz", allow_pickle=True)).iloc[
        date_index : date_index + 60
    ]
    before_data = before_data.drop("Sentiment", axis=1)
    print(before_data.head())
    before_data.loc[:, "UnixTime"] = before_data["UnixTime"].apply(
        lambda x: datetime.datetime.fromtimestamp(x)
    )
    output_df.loc[:, "UnixTime"] = output_df["UnixTime"].apply(
        lambda x: datetime.datetime.fromtimestamp(x)
    )
    return before_data, output_df


def main():
    min_date = datetime.datetime(2024, 7, 1)
    max_date = datetime.datetime(2025, 6, 1)
    firms = [
        "AAPL",
        "AMD",
        "AVGO",
        "CRM",
        "CSCO",
        "IBM",
        "MSFT",
        "NVDA",
        "ORCL",
        "PLTR",
    ]
    with gr.Blocks() as demo:
        gr.Markdown("## Firm Forecast Dashboard")

        with gr.Row():
            firm_input = gr.Dropdown(choices=firms, label="Select Firm")
            start_date_input = gr.DateTime(
                label="Start Date", value=datetime.datetime(2025, 1, 1)
            )

        line_plot = gr.LinePlot(
            x="Time",
            y="Close",
            color="source",
            title="Stock Price Prediction",
            tooltip=["Time", "Close", "Type"],
            height=450,
            width=700,
        )

        # --- Input handling and clamping ---
        def handle_inputs(firm, start_date: float):
            # Clamp to date range
            start_date: datetime.datetime = remove_nyse_days(
                datetime.datetime.fromtimestamp(start_date), -60
            )
            if start_date < min_date:
                start_date = min_date
            elif start_date > max_date:
                start_date = max_date

            hist, pred = predict_model(start_date, firm)
            hist["UnixTime"] = hist["UnixTime"].dt.to_pydatetime()
            pred["UnixTime"] = pred["UnixTime"].dt.to_pydatetime()
            hist.rename(columns={"UnixTime": "Time"}, inplace=True)
            pred.rename(columns={"UnixTime": "Time"}, inplace=True)
            print(type(hist.iloc[0]["Time"]))
            print(type(pred.iloc[0]["Time"]))
            hist["source"] = "Historical"
            pred["source"] = "Predicted"
            final = pd.concat([hist, pred])
            return final

        submit_btn = gr.Button("Predict Stock Price")
        submit_btn.click(
            fn=handle_inputs, inputs=[firm_input, start_date_input], outputs=line_plot
        )

    demo.launch()


if __name__ == "__main__":
    main()
