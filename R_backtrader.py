import time
import backtrader as bt
import numpy as np
import lightgbm as lgb
import polars as pl
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd

import D_processdata

class AITradingStrategy(bt.Strategy):
    params = (
        ('lgb_model_path', 'lgb_model.txt'),  # LightGBM 模型路径
        ('transformer_model_path', 'transformer_model.pth'),  # Transformer 模型路径
        ('prediction_window', 12),  # 预测窗口
    )

    def __init__(self):
        # 加载模型
        self.lgb_model = lgb.Booster(model_file=self.params.lgb_model_path)
        #self.transformer_model = torch.load(self.params.transformer_model_path)
        #self.transformer_model.eval()

        # 数据缓冲区
        self.data_buffer = []
        self.predictions = []  # 存储预测值
        self.actual_prices = []  # 存储实际价格
        self.dates = []  # 存储时间戳

    def next(self):
        # 收集最新数据
        new_row = [getattr(self.datas[0], feat)[0] for feat in D_processdata.features]
        self.data_buffer.append(new_row)

        if len(self.data_buffer) >= 24:  # 输入需要24个时间步
            # LightGBM 预测
            lgb_input = np.array(self.data_buffer[-24:]).reshape(1, -1)
            lgb_pred = self.lgb_model.predict(lgb_input, predict_disable_shape_check=True)

            # Transformer 预测
            #transformer_input = torch.tensor([self.data_buffer[-24:]], dtype=torch.float32)
            #transformer_pred = self.transformer_model(transformer_input).item()

            # 综合预测结果
            #final_pred = 0.5 * lgb_pred + 0.5 * transformer_pred
            final_pred = lgb_pred

            # 记录预测值和当前时间
            self.predictions.append(final_pred)
            self.dates.append(self.datas[0].datetime.datetime(0))  # 记录当前时间
            self.actual_prices.append(self.datas[0].close[0])  # 记录实际的close价格

            # 交易逻辑
            if final_pred > 0.01:  # 预测涨幅超过1%
                self.order_target_percent(target=0.95)  # 95%仓位
            elif final_pred > 0.05:  # 预测涨幅超过5%
                self.order_target_percent(target=0.99)  # 99%仓位
            elif final_pred < -0.0005:  # 预测跌幅超过0.5%
                self.order_target_percent(target=0.05)  # 保留5%仓位
            elif final_pred < -0.01:  # 预测跌幅超过1%
                self.order_target_percent(target=0)  # 跑


class CustomPandasData(bt.feeds.PandasData):
    # 添加额外的指标列
    lines = ('close', 'rsi', 'macd', 'atr', 'log_return', 'ma20', 'volume_change_rate', 'close_lag_1',
             'close_lag_2', 'close_lag_3', 'close_lag_6', 'close_lag_12', 'future_return')
    params = (
        ('future_return', 17),
        ('close', 4),
        ('rsi', 6),
        ('macd', 7),
        ('atr', 8),
        ('log_return', 9),
        ('ma20', 10),
        ('volume_change_rate', 11),
        ('close_lag_1', 12),
        ('close_lag_2', 13),
        ('close_lag_3', 14),
        ('close_lag_6', 15),
        ('close_lag_12', 16)
    )


# 创建回测引擎
cerebro = bt.Cerebro()

# 添加数据
endTime = int(time.time()*1000)
raw_data = D_processdata.fetch_binance_data(symbol='BTCUSDT', interval='5m', endTime=endTime)
processed_data = D_processdata.create_features(raw_data)
# 添加滞后特征
for lag in [1, 2, 3, 6, 12]:
    processed_data = processed_data.with_columns(
        pl.col('close').shift(lag).alias(f'close_lag_{lag}')
    )
processed_data = processed_data.with_columns(
    (pl.col('close').shift(-D_processdata.target_horizon * 2) / pl.col('close') - 1)  # 增加时间窗口
    .alias('future_return')
).drop_nulls().drop_nans()


scaler = StandardScaler()
scaled_features = scaler.fit_transform(processed_data[D_processdata.features].to_pandas())
processed_data = processed_data.with_columns(
    [pl.Series(D_processdata.features[i], scaled_features[:, i]) for i in range(len(D_processdata.features))]
)

# 调整目标变量范围
processed_data = processed_data.with_columns(
    (pl.col('future_return') * 100).alias('future_return')  # 放大100倍
)
processed_data = processed_data.to_pandas()
data = CustomPandasData(dataname=processed_data, datetime=0)

cerebro.adddata(data)

# 添加策略
cerebro.addstrategy(AITradingStrategy)

# 设置初始资金
cerebro.broker.set_cash(100000)

# 运行回测
print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
cerebro.run()
print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

# 绘制结果
cerebro.plot()
