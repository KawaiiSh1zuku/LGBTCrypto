import polars as pl
import binance
import numpy as np
import talib as ta
import time
from tqdm import trange
# 标准化特征
from sklearn.preprocessing import StandardScaler

target_horizon = 12  # 5分钟*12=1小时
features = ['rsi', 'macd', 'atr', 'log_return', 'ma20', 'volume_change_rate']
features.extend([f'close_lag_{lag}' for lag in [1, 2, 3, 6, 12]])


# 1. 实时数据获取
def fetch_binance_data(symbol='BTCUSDT', interval='5m', limit=1000, endTime=None):
    klines = binance.klines(symbol=symbol, interval=interval, limit=limit, endTime=endTime)
    df = pl.DataFrame({
        "timestamp": [str(x['openTime']) for x in klines],
        "open": [float(x['open']) for x in klines],
        "high": [float(x['high']) for x in klines],
        "low": [float(x['low']) for x in klines],
        "close": [float(x['close']) for x in klines],
        "volume": [float(x['volume']) for x in klines]
    })
    return df.with_columns(
        pl.col("timestamp").cast(pl.Int64()).cast(pl.Datetime("ms"))
    )


# 2. 特征工程
def create_features(df: pl.DataFrame):
    # 计算技术指标
    closes = df['close'].to_numpy()
    return df.with_columns([
        pl.Series(ta.RSI(closes, timeperiod=14)).alias('rsi'),
        pl.Series(ta.MACD(closes)[0]).alias('macd'),
        pl.Series(ta.ATR(df['high'].to_numpy(),
                         df['low'].to_numpy(),
                         closes, timeperiod=14)).alias('atr'),
        np.log(df['close'] / df['close'].shift(1)).alias('log_return'),
        df['close'].rolling_mean(window_size=20).alias('ma20'),
        ((df['volume'] - df['volume'].shift(1)) / df['volume'].shift(1))
        .alias('volume_change_rate')
    ]).drop_nans()

def main():
    endTime = int(time.time()*1000)
    raw_data = fetch_binance_data(endTime=endTime)
    for i in trange(0, 1000):
        try:
            endTime = raw_data[0]['timestamp'].cast(pl.Int64)
        except:
            endTime = int(time.time()*1000)
        df = fetch_binance_data(endTime=endTime)
        raw_data = pl.concat([raw_data, df], rechunk=True)

    processed_data = create_features(raw_data)

    # 添加滞后特征
    for lag in [1, 2, 3, 6, 12]:
        processed_data = processed_data.with_columns(
            pl.col('close').shift(lag).alias(f'close_lag_{lag}')
        )
    processed_data = processed_data.with_columns(
        (pl.col('close').shift(-target_horizon * 2) / pl.col('close') - 1)  # 增加时间窗口
        .alias('future_return')
    ).drop_nulls().drop_nans()


    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(processed_data[features].to_pandas())

    processed_data = processed_data.with_columns(
        [pl.Series(features[i], scaled_features[:, i]) for i in range(len(features))]
    )

    # 调整目标变量范围
    processed_data = processed_data.with_columns(
        (pl.col('future_return') * 100).alias('future_return')  # 放大100倍
    )
    processed_data.write_csv('processed.csv')

if "__name__" == "__main__":
    main()