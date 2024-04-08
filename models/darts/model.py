import darts
import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union
from darts.metrics import mse, mae
from darts.utils.likelihood_models import QuantileRegression
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger


class DartsModel(ABC):
    def __init__(
            self,
            timeframe_size: int,
            prediction_len: int,
            quantiles: List[float],
            model_name: str,
            work_dir: str,
            seed: int,
            device_id: int,
            **params
    ):
        self.timeframe_size = timeframe_size
        self.prediction_len = prediction_len
        self.model_name = model_name
        self.work_dir = work_dir
        self.seed = seed
        self.device_id = device_id
        self.params = params

        self.loss_fn = QuantileRegression(quantiles=quantiles)
        self._model = None

    @property
    def model(self):
        return self._model

    @abstractmethod
    def _build(self, learning_rate: float, batch_size: int, pl_trainer_dict: Dict[str, Any]) -> Any:
        pass

    def build(self, learning_rate: float, batch_size: int, early_stopping_patience: int, tensorboard_dir: str):
        pl_trainer_dict = self._get_pl_trainer_dict(
            early_stopping_patience=early_stopping_patience,
            tensorboard_dir=tensorboard_dir
        )
        self._build(learning_rate=learning_rate, batch_size=batch_size, pl_trainer_dict=pl_trainer_dict)

        if self._model is None:
            raise NotImplementedError('Model has not been built.')

    def _get_pl_trainer_dict(self, early_stopping_patience: int, tensorboard_dir: str) -> Dict[str, Any]:
        early_stopper_callback = EarlyStopping(
            monitor='val_loss',
            patience=early_stopping_patience,
            min_delta=0.001,
            mode='min',
        )
        tensorboard_logger = TensorBoardLogger(tensorboard_dir)
        pl_trainer_kwargs = {
            'callbacks': [early_stopper_callback],
            'logger': tensorboard_logger
        }

        if self.device_id > -1:
            pl_trainer_kwargs['accelerator'] = 'gpu'
            pl_trainer_kwargs['devices'] = self.device_id

        return pl_trainer_kwargs

    def load_checkpoint(self, model_name: str, work_dir: str):
        self._model = self._model.load_from_checkpoint(
            model_name=model_name,
            best=True,
            work_dir=work_dir
        )

    def train_model(
            self,
            x_train: darts.TimeSeries,
            y_train: darts.TimeSeries,
            x_eval: darts.TimeSeries,
            y_eval: darts.TimeSeries,
            epochs: int
    ):
        if self._model is None:
            raise RuntimeError('Model is None. Build model first.')

        self._model.fit(
            series=y_train,
            past_covariates=x_train,
            val_series=y_eval,
            val_past_covariates=x_eval,
            epochs=epochs
        )

    def eval_model(self, x_test: darts.TimeSeries, y_test: darts.TimeSeries) -> (float, float):
        if self._model is None:
            raise RuntimeError('Model is None. Build model first.')

        return self._model.backtest(
            series=y_test,
            past_covariates=x_test,
            forecast_horizon=self.prediction_len,
            stride=1,
            last_points_only=False,
            retrain=False,
            metric=[mse, mae],
            verbose=False
        )

    def predict(
            self,
            x: Union[List[darts.TimeSeries], darts.TimeSeries],
            y: Union[List[darts.TimeSeries], darts.TimeSeries]
    ) -> darts.TimeSeries:
        return self._model.historical_forecasts(
            series=y,
            past_covariates=x,
            start=None,
            retrain=False,
            forecast_horizon=1,
            stride=1
        )
