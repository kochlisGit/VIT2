import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import ADXIndicator, CCIIndicator, EMAIndicator
from ta.volume import VolumeWeightedAveragePrice


class DatasetLoader:
    def __init__(self, dataset_directory: str):
        self._dataset_directory = dataset_directory

        self._volume_columns = ['volume usdt', 'volume usd', 'volfrom', 'base_volume']
        self._candlestick_columns = ['date', 'open', 'high', 'low', 'close', 'volume']

    def _load_daily_dataframe(self, exchange: str, symbol: str, noise_percentage: float = 0.0) -> pd.DataFrame:
        def extract_candlesticks(df: pd.DataFrame) -> pd.DataFrame:
            df.columns = [col.lower() for col in df.columns]
            for col in self._volume_columns:
                if col in df.columns:
                    df.rename(columns={col: 'volume'}, inplace=True)
                    return df[self._candlestick_columns]

            assert RuntimeError(
                f'Expected at least one of these volume columns in dataframe: {self._volume_columns}. Got: {df.columns}'
            )

        dataset_filepath = f'{self._dataset_directory}/{exchange}_{symbol}_d.csv'
        df = pd.read_csv(dataset_filepath, skiprows=1)
        df = extract_candlesticks(df=df)

        if noise_percentage > 0:
            noise_factor = 1 + np.random.normal(loc=0, scale=noise_percentage)
            df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']]*noise_factor

        df.insert(loc=0, column='exchange', value=exchange)
        df.insert(loc=1, column='symbol', value=symbol)
        df['date'] = pd.to_datetime(df['date'])
        return df.dropna().sort_values(by='date', ascending=True, ignore_index=True)

    def _compute_candlestick_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df['ema'] = EMAIndicator(close=df['close']).ema_indicator()
        df['vwap'] = VolumeWeightedAveragePrice(high=df['high'], low=df['low'], close=df['close'], volume=df['volume']).volume_weighted_average_price()
        df['rsi'] = RSIIndicator(close=df['close']).rsi()
        df['cci'] = CCIIndicator(high=df['high'], low=df['low'], close=df['close']).cci()
        df['adx'] = ADXIndicator(high=df['high'], low=df['low'], close=df['close']).adx()

        if 'day' not in df.columns:
            df['day'] = df['date'].dt.dayofweek
        return df

    def _compute_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        df['targets'] = np.log(df['close'].shift(-1)/df['close']).shift(1)
        return df

    def _validate_dataframe(self, df: pd.DataFrame):
        expected_columns = ['exchange', 'symbol'] + self._candlestick_columns

        for col in expected_columns:
            assert col in df.columns, f'Expected {col} in dataframe, got {df.columns}'

        if df['date'].isna().any().any():
            raise RuntimeError(f'Dataframe should not contain nan values, found nan in columns: {df.isna()}')
        if not df['date'].is_monotonic_increasing:
            raise RuntimeError('Dataframe is expected to be sorted in ascending order')
        if df['date'].duplicated().any():
            raise RuntimeError('Dataframe should not contain duplicated dates')

    def load_datasets(self, dataset_configs: list[dict], noise_percentage: float = 0.0) -> pd.DataFrame:
        dfs_list = []
        for config in dataset_configs:
            df = self._load_daily_dataframe(
                exchange=config['exchange'],
                symbol=config['symbol'],
                noise_percentage=noise_percentage)
            df = self._compute_candlestick_features(df=df)
            df = self._compute_targets(df=df)
            df = df.dropna().reset_index(drop=True)
            df = df[~df.isin([np.inf, -np.inf]).any(axis=1)]
            self._validate_dataframe(df=df)

            dfs_list.append(df)
        return pd.concat(dfs_list, axis=0)
