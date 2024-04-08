import random
import darts
import numpy as np
import pandas as pd
import torch
from typing import Any, List
from pyts.image import GramianAngularField
from sklearn.preprocessing import MinMaxScaler


def config_split_train_test(configs: list[dict[str, str]], test_size: float) -> (list[dict[str, str]], list[dict[str, str]]):
    random.shuffle(configs)
    split_idx = int((1-test_size)*len(configs))
    train_configs = configs[:split_idx]
    test_configs = configs[split_idx:]
    return train_configs, test_configs


def construct_image_dataset(df: pd.DataFrame, timeframe_size: int) -> (np.ndarray, np.ndarray):
    def construct_group_dataset(group_df: pd.DataFrame):
        def apply_gaf() -> np.ndarray:
            time_series = group_df.drop(columns=['exchange', 'symbol', 'date']).to_numpy(dtype=np.float64)
            timeframes = np.float64([time_series[i: i + timeframe_size] for i in range(time_series.shape[0] - timeframe_size)])
            channels = [
                GramianAngularField(method='difference', sample_range=(0, 1)).fit_transform(timeframes[:, :, i])
                for i in range(timeframes.shape[2])
            ]
            return np.stack(channels, axis=1)

        def generate_targets() -> np.ndarray:
            return np.expand_dims(group_df['targets'].to_numpy(dtype=np.float64)[timeframe_size:], axis=-1)

        images = apply_gaf()
        targets = generate_targets()

        assert images.shape[0] == targets.shape[0], f'Inputs-Targets size mismatch: {images.shape[0]} vs {targets.shape[0]}'

        return images, targets

    x_list = []
    y_list = []
    for _, group_dataframe in df.groupby(['exchange', 'symbol']):
        if group_dataframe.shape[0] > timeframe_size:
            x, y = construct_group_dataset(group_df=group_dataframe)
            x_list.append(x)
            y_list.append(y)
    return np.vstack(x_list), np.vstack(y_list)


def construct_dataloader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool):
    assert x.shape[0] == y.shape[0], f'Inputs-Targets size mismatch: {x.shape[0]} vs {y.shape[0]}'

    dataset = torch.utils.data.TensorDataset(torch.Tensor(x), torch.Tensor(y))
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def construct_timeseries_dataset(
        df: pd.DataFrame,
        timeframe_size: int,
        scale_data: bool
) -> (List[darts.TimeSeries], List[darts.TimeSeries], Any):
    targets = []
    past_cov = []
    features = [col for col in df.columns if col != 'exchange' and col != 'symbol' and col != 'date']

    for _, group_df in df.groupby(['exchange', 'symbol']):
        if group_df.shape[0] > timeframe_size:
            group_df['time_idx'] = group_df.index.astype(int)

            if scale_data:
                scaler = MinMaxScaler()
                cov_values = group_df[features]
                cov_values = scaler.fit_transform(cov_values)
                group_df[features] = cov_values
            else:
                scaler = None

            past_cov.append(darts.TimeSeries.from_dataframe(
                df=group_df,
                time_col='time_idx',
                value_cols=features
            ))
            targets.append(darts.TimeSeries.from_dataframe(
                df=group_df,
                time_col='time_idx',
                value_cols=['targets']
            ))
        else:
            exchange = group_df.iloc[0]['exchange']
            symbol = group_df.iloc[0]['symbol']
            print(f'Warning: dataset: {exchange}-{symbol} has only {group_df.shape[0]} samples. At least {timeframe_size} are required. Skipping that dataset.')
    return past_cov, targets, scaler
