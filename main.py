import os
import time
import datetime
import threading
import requests
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

# ----------------------------
# 0) Fetch with Retry ve Headers Ekleme
# ----------------------------

DEFAULT_HEADERS = {
    'User-Agent': 'Mozilla/5.0'
}

def fetch_data_with_retry(url, params=None, headers=None, retries=3, timeout=10):
    """
    URL'ye GET isteği atar, headers ve params kullanır. Hata alırsa retries kadar tekrar dener.
    """
    if headers is None:
        headers = DEFAULT_HEADERS
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"[Retry {attempt}/{retries}] Hata: {e} | URL: {url} | Params: {params}")
            if attempt < retries:
                time.sleep(2)
            else:
                print(f"[ERROR] Veri alınamadı: {url}")
                return None

# ----------------------------
# 1) Ortam Değişkenlerini Yükle
# ----------------------------
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
if not TELEGRAM_TOKEN or not CHAT_ID:
    raise RuntimeError("Lütfen .env dosyasına TELEGRAM_TOKEN ve CHAT_ID ekleyin.")

# ----------------------------
# 2) Sabitler ve Ayarlar
# ----------------------------
BINANCE_FUTURES_BASE = "https://fapi.binance.com"
SYMBOLS_ENDPOINT = "/fapi/v1/ticker/price"      # Tüm futures sembollerini çeker
KLINES_ENDPOINT = "/fapi/v1/klines"             # 1 dakikalık historial veriyi alır

INTERVAL = "15m"                                 # 1 dakikalık zaman dilimi
RSI_PERIOD = 20                                 # RSI periyodu
RSI_SMA_PERIOD = 7                             # RSI değeri için SMA periyodu
STOCH_K_PERIOD = 20                             # Stoch %K periyodu
STOCH_D_PERIOD = 2                       # Stoch %D periyodu
STOCH_SMOOTH = 2                                # Stoch %K smoothing periyodu
STOCH2_K_PERIOD = 5                            # Stoch %K periyodu
STOCH2_D_PERIOD = 3                       # Stoch %D periyodu
STOCH2_SMOOTH = 2                               # Stoch %K smoothing 
OBV_SMA_PERIOD = 20
OBV2_SMA_PERIOD = 15
# OBV için SMA periyodu
EMA_PERIOD = 20   
EMA2_PERIOD = 40  
EMA3_PERIOD = 60  
EMA4_PERIOD = 80
EMA5_PERIOD = 45   
EMA6_PERIOD = 15
# EMA periyodu
THREAD_COUNT = 20                               # Thread sayısı (PoolExecutor)

# Al-Sat eşik değerleri
RSI_OVERSOLD = 55
RSI_OVERBOUGHT = 45
STOCH_OVERSOLD = 30
STOCH_OVERBOUGHT = 70

# ----------------------------
# 3) Yardımcı İndikatör Fonksiyonları
# ----------------------------
def calculate_rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0.0)).fillna(0.0)
    loss = (-delta.where(delta < 0, 0.0)).fillna(0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    avg_gain = avg_gain.copy()
    avg_loss = avg_loss.copy()
    for i in range(period, len(series)):
        if i == period:
            continue
        avg_gain.iloc[i] = (avg_gain.iloc[i - 1] * (period - 1) + gain.iloc[i]) / period
        avg_loss.iloc[i] = (avg_loss.iloc[i - 1] * (period - 1) + loss.iloc[i]) / period
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period, min_periods=period).mean()

def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def calculate_ema2(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def calculate_ema3(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def calculate_ema4(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def calculate_ema5(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def calculate_ema6(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()



def calculate_stochastic(df: pd.DataFrame, k_period: int, d_period: int, smooth: int) -> pd.DataFrame:
    low_min = df["low"].rolling(window=k_period, min_periods=k_period).min()
    high_max = df["high"].rolling(window=k_period, min_periods=k_period).max()
    stoch_k = 100 * (df["close"] - low_min) / (high_max - low_min)
    k_smooth = stoch_k.rolling(window=smooth, min_periods=smooth).mean()
    d_val = k_smooth.rolling(window=d_period, min_periods=d_period).mean()
    return pd.DataFrame({
        "%K_smooth": k_smooth,
        "%D": d_val

    }, index=df.index)





def calculate_obv(df: pd.DataFrame) -> pd.DataFrame:
    obv = [0]
    for i in range(1, len(df)):
        if df["close"].iloc[i] > df["close"].iloc[i - 1]:
            obv.append(obv[-1] + df["volume"].iloc[i])
        elif df["close"].iloc[i] < df["close"].iloc[i - 1]:
            obv.append(obv[-1] - df["volume"].iloc[i])
        else:
            obv.append(obv[-1])
    obv_series = pd.Series(obv, index=df.index)
    obv_sma = obv_series.rolling(window=OBV_SMA_PERIOD, min_periods=OBV_SMA_PERIOD).mean()
    obv2_sma = obv_series.rolling(window=OBV2_SMA_PERIOD, min_periods=OBV2_SMA_PERIOD).mean()

    return pd.DataFrame({
        "OBV": obv_series,
        "OBV_SMA": obv_sma,
        "OBV2_SMA": obv2_sma


    }, index=df.index)

# ----------------------------
# 4) Binance’dan Veri Çekme (Headers & Retry)
# ----------------------------
def get_all_futures_symbols() -> list:
    """
    Binance Futures'taki public tüm sembolleri çeker.
    """
    url = BINANCE_FUTURES_BASE + SYMBOLS_ENDPOINT
    data = fetch_data_with_retry(url, headers=DEFAULT_HEADERS)
    if not data:
        return []
    symbols = [item["symbol"] for item in data]
    print(f"[INFO] Toplam sembol sayısı: {len(symbols)}")
    return symbols

def get_klines(symbol: str, interval: str, limit: int = 500) -> pd.DataFrame:
    """
    Belirli bir sembol için 1 dakikalık klines verisini alır (limit bar).
    """
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    url = BINANCE_FUTURES_BASE + KLINES_ENDPOINT
    data = fetch_data_with_retry(url, params=params, headers=DEFAULT_HEADERS)
    if not data:
        return None
    df = pd.DataFrame(data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "num_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
    ])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df.set_index("open_time", inplace=True)
    return df[["open", "high", "low", "close", "volume"]]

# ----------------------------
# 5) Sinyal Üretme Mantığı (Aynı) ve Console Log Ekleme
# ----------------------------
def decide_signal(indics: dict) -> str:
    rsi = indics["rsi"]
    rsi_sma = indics["rsi_sma"]
    stoch_k = indics["stoch_k"]
    stoch_d = indics["stoch_d"]
    stoch2_k = indics["stoch2_k"]
    stoch2_d = indics["stoch2_d"]

    obv = indics["obv"]
    obv_sma = indics["obv_sma"]
    obv2_sma = indics["obv2_sma"]
    ema = indics["ema"]
    ema2 = indics["ema2"]
    ema3 = indics["ema3"]
    ema4 = indics["ema4"]
    ema5 = indics["ema5"]
    ema6 = indics["ema6"]

    close_price = indics["close"]
    

    # BUY şartı
    if (
        (rsi is not None and rsi >51) and 
        (stoch_k is not None and stoch_k>stoch_d ) and         
        (obv is not None and obv>obv_sma  ) and 
        close_price>ema2 and ema>ema2>ema3
        
        ):
        return "BUY"

    # SELL şartı
    if (
        (rsi is not None and rsi < 49) and 
        (stoch_k is not None and  stoch_d>stoch_k ) and
        (obv is not None and obv < obv_sma ) and
        close_price<ema2 and ema<ema2<ema3
        
    ):
        return "SELL"

    return ""

def process_symbol(symbol: str, result_list: list, timestamp: pd.Timestamp):
    try:
        df = get_klines(symbol, INTERVAL, limit=500)
        if df is None or len(df) < max(RSI_PERIOD, STOCH_K_PERIOD + STOCH_SMOOTH, OBV_SMA_PERIOD, EMA6_PERIOD) + 22:
            print(f"[WARNING] Yetersiz veri: {symbol}")
            return

        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        # İndikatör hesaplamaları
        rsi_series = calculate_rsi(close, RSI_PERIOD)
        rsi_sma_series = calculate_sma(rsi_series, RSI_SMA_PERIOD)

        stoch_df = calculate_stochastic(df, STOCH_K_PERIOD, STOCH_D_PERIOD, STOCH_SMOOTH)
        stoch2_df = calculate_stochastic(df, STOCH2_K_PERIOD, STOCH2_D_PERIOD, STOCH2_SMOOTH)

        obv_df = calculate_obv(df)

        ema_series = calculate_ema(close, EMA_PERIOD)
        ema2_series = calculate_ema2(close, EMA2_PERIOD)
        ema3_series = calculate_ema3(close, EMA3_PERIOD)
        ema4_series = calculate_ema4(close, EMA4_PERIOD)
        ema5_series = calculate_ema5(close, EMA5_PERIOD)
        ema6_series = calculate_ema6(close, EMA6_PERIOD)


        indics = {
            "rsi": float(rsi_series.iloc[-2]) if not np.isnan(rsi_series.iloc[-2]) else None,
            "rsi_sma": float(rsi_sma_series.iloc[-2]) if not np.isnan(rsi_sma_series.iloc[-2]) else None,
            "stoch_k": float(stoch_df["%K_smooth"].iloc[-1]) if not np.isnan(stoch_df["%K_smooth"].iloc[-1]) else None,
            "stoch_d": float(stoch_df["%D"].iloc[-1]) if not np.isnan(stoch_df["%D"].iloc[-2]) else None,
            "stoch2_k": float(stoch2_df["%K_smooth"].iloc[-2]) if not np.isnan(stoch2_df["%K_smooth"].iloc[-2]) else None,
            "stoch2_d": float(stoch2_df["%D"].iloc[-2]) if not np.isnan(stoch2_df["%D"].iloc[-2]) else None,

            "obv": float(obv_df["OBV"].iloc[-2]) if not np.isnan(obv_df["OBV"].iloc[-2]) else None,
            "obv_sma": float(obv_df["OBV_SMA"].iloc[-2]) if not np.isnan(obv_df["OBV_SMA"].iloc[-2]) else None,
            "obv2_sma": float(obv_df["OBV2_SMA"].iloc[-1]) if not np.isnan(obv_df["OBV2_SMA"].iloc[-1]) else None,

            "ema": float(ema_series.iloc[-2]) if not np.isnan(ema_series.iloc[-2]) else None,
            "ema2": float(ema2_series.iloc[-2]) if not np.isnan(ema2_series.iloc[-2]) else None,
            "ema3": float(ema3_series.iloc[-1]) if not np.isnan(ema3_series.iloc[-1]) else None,
            "ema4": float(ema4_series.iloc[-1]) if not np.isnan(ema4_series.iloc[-1]) else None,
            "ema5": float(ema5_series.iloc[-1]) if not np.isnan(ema5_series.iloc[-1]) else None,
            "ema6": float(ema6_series.iloc[-1]) if not np.isnan(ema6_series.iloc[-1]) else None,

            "close": float(close.iloc[-2]
           
            )}
        

        signal = decide_signal(indics)
        if signal:
            if len(close) >= 460:
                price_2h_ago = close.iloc[-60]
                price_now = close.iloc[-2]
                change_2h = (price_now - price_2h_ago) / price_2h_ago * 100
            else:
                change_2h = 0.0

            result_list.append((symbol, signal, round(change_2h, 2)))
            print(f"[SIGNAL] {symbol} | {signal} | 2h Δ: {round(change_2h,2)}%")
        else:
            print(f"[NO SIGNAL] {symbol}")
    except Exception as e:
        print(f"[ERROR] process_symbol {symbol}: {e}")
        return

# ----------------------------
# 6) Telegram Mesaj Gönderme
# ----------------------------
def send_telegram_message(message: str):
    """
    Telegram bot API ile mesaj gönderir.
    """
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    try:
        resp = requests.post(url, data=payload, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        print(f"[ERROR] Telegram mesaj gönderilirken hata: {e}")

# ----------------------------
# 7) Ana Tarama Döngüsü
# ----------------------------
def scan_all_symbols():
    symbols = get_all_futures_symbols()
    now = pd.Timestamp.utcnow()
    manager_list = []

    with ThreadPoolExecutor(max_workers=THREAD_COUNT) as executor:
        futures = []
        for sym in symbols:
            futures.append(executor.submit(process_symbol, sym, manager_list, now))
        for f in futures:
            f.result()

    buy_list = [(s, change) for (s, signal, change) in manager_list if signal == "BUY"]
    sell_list = [(s, change) for (s, signal, change) in manager_list if signal == "SELL"]

    buy_list = sorted(buy_list, key=lambda x: x[1])

    sell_list = sorted(sell_list, key=lambda x: x[1], reverse=True) 
    buy_list = buy_list[:10]
    sell_list = sell_list[:10]

    lines = []
    lines.append(f"*15dk Tarama Zamanı:* `{now.strftime('%Y-%m-%d %H:%M:%S')} UTC`")
    lines.append("")
    if buy_list:
        lines.append("*🟢 BUY Sinyalleri (En yüksekten en düşüğe)*")
        for sym, chg in buy_list:
            lines.append(f"`{sym}`  |  2h Δ: `{chg}%`")



    lines.append("")
    if sell_list:
        lines.append("*🔴 SELL Sinyalleri (En düşükten en yükseğe)*")
        for sym, chg in sell_list:
            lines.append(f"`{sym}`  |  2h Δ: `{chg}%`")

    if buy_list or sell_list:
        message = "\n".join(lines)
        send_telegram_message(message)
    else:    
        pass

def run_scheduler():
    while True:
        now = datetime.datetime.utcnow()
        seconds_to_next_minute = 60 - now.second
        time.sleep(seconds_to_next_minute)
        scan_all_symbols()

if __name__ == "__main__":
    print("Bot başlatılıyor... Her dakika başında tarama yapılacaktır.")
    run_scheduler()
