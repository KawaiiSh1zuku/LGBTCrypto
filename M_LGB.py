import lightgbm as lgb
import optuna
import polars as pl
from sklearn.metrics import mean_absolute_error

# 预测未来1小时的收益率
target_horizon = 12  # 5分钟*12=1小时
features = ['rsi', 'macd', 'atr', 'log_return', 'ma20', 'volume_change_rate']
features.extend([f'close_lag_{lag}' for lag in [1, 2, 3, 6, 12]])

# 1. 从 CSV 加载数据
processed_data = pl.read_csv('processed.csv')


def split_data(data, test_size=0.2):
    """
    按时间顺序分割数据集
    :param data: 输入数据（Polars DataFrame）
    :param test_size: 验证集比例
    :return: 训练集和验证集
    """
    split_idx = int(len(data) * (1 - test_size))
    train_data = data[:split_idx]
    valid_data = data[split_idx:]
    return train_data, valid_data


train_data, valid_data = split_data(processed_data, test_size=0.2)  # 自定义时间序列分割

train_labels = train_data['future_return'].to_pandas().astype(float)
valid_labels = valid_data['future_return'].to_pandas().astype(float)
train_data = train_data.to_pandas()
valid_data = valid_data.to_pandas()


# 3. Optuna自动调参
def objective(trial):
    params = {
        'objective': 'regression',
        'metric': 'mae',
        'boosting_type': trial.suggest_categorical('boosting', ['gbdt', 'dart']),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'min_data_in_leaf': trial.suggest_int('min_data', 10, 50),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.7, 1.0),
        'lambda_l1': trial.suggest_float('lambda_l1', 0.0, 1.0),
        'lambda_l2': trial.suggest_float('lambda_l2', 0.0, 1.0),
    }

    train_dataset = lgb.Dataset(train_data[features], train_labels)
    valid_dataset = lgb.Dataset(valid_data[features], valid_labels)

    model = lgb.train(
        params,
        train_dataset,
        valid_sets=[valid_dataset],
    )
    # 模型预测
    preds = model.predict(valid_data[features])
    mae = mean_absolute_error(valid_labels, preds)
    return mae


# 训练执行
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

best_model = lgb.train(study.best_params, lgb.Dataset(valid_data[features], valid_labels))
best_model.save_model('lgb_model.txt')
