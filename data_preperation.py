import oandapyV20
from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import ta  # pip install ta

def fetch_oanda_data(instrument, start_date, end_date, granularity="D", access_token=None):
    """
    Fetch historical OHLC data from OANDA for the given instrument.
    """
    client = API(access_token=access_token)
    start_str = start_date.strftime("%Y-%m-%dT00:00:00Z")
    end_str = end_date.strftime("%Y-%m-%dT00:00:00Z")
    
    params = {
        "from": start_str,
        "to": end_str,
        "granularity": granularity,
        "price": "M"  # Use mid prices
    }
    
    r = instruments.InstrumentsCandles(instrument=instrument, params=params)
    client.request(r)
    candles = r.response.get('candles', [])
    
    data = []
    for candle in candles:
        if candle["complete"]:
            time = candle["time"]
            o = float(candle["mid"]["o"])
            h = float(candle["mid"]["h"])
            l = float(candle["mid"]["l"])
            c = float(candle["mid"]["c"])
            data.append([time, o, h, l, c])
    
    df = pd.DataFrame(data, columns=["Date", "Open", "High", "Low", "Close"])
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    return df

def compute_technical_indicators(df):
    """
    Compute key technical indicators and add them as new columns.
    """
    # 20-day Simple Moving Average
    df['SMA20'] = df['Close'].rolling(window=20).mean()

    # 14-day Relative Strength Index
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)

    # MACD and MACD Signal
    macd = ta.trend.MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()

    # Bollinger Bands (20-day, 2 standard deviations)
    bollinger = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
    df['Bollinger_High'] = bollinger.bollinger_hband()
    df['Bollinger_Low'] = bollinger.bollinger_lband()
    
    return df

def prepare_features_and_target(df):
    """
    Create the regression target (next-day return) and prepare the feature set.
    """
    # Calculate next day's return
    df['Return'] = df['Close'].pct_change().shift(-1)
    df = df.dropna()  # Drop rows with NaN values from rolling calculations and shift
    
    features = df[['SMA20', 'RSI', 'MACD', 'MACD_signal', 'Bollinger_High', 'Bollinger_Low']]
    target = df['Return']
    return features, target

# --- Main Execution ---
if __name__ == "__main__":
    # Setup for data collection
    access_token = "YOUR_ACCESS_TOKEN"  # Replace with your OANDA API token
    instrument = "XAU_USD"  # Forex pair for Gold

    # Set end_date to two days before now
    end_date = datetime.utcnow() - timedelta(days=2)
    # Use 5 years of data ending at end_date
    start_date = end_date - timedelta(days=5*365)

    # Fetch data from OANDA
    df_raw = fetch_oanda_data(instrument, start_date, end_date, granularity="D", access_token=access_token)
    print("Raw Data Head:")
    print(df_raw.head())

    # Compute technical indicators
    df_with_indicators = compute_technical_indicators(df_raw.copy())
    print("\nData with Indicators Head:")
    print(df_with_indicators.head())

    # Prepare features and target
    features, target = prepare_features_and_target(df_with_indicators.copy())
    print("\nFeatures Head:")
    print(features.head())
    print("\nTarget (Next Day Return) Head:")
    print(target.head())
