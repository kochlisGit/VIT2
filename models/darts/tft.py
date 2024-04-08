from typing import Any, Dict, List
from darts.models import TFTModel
from models.darts.model import DartsModel


class TFT(DartsModel):
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
        super().__init__(
            timeframe_size=timeframe_size,
            prediction_len=prediction_len,
            quantiles=quantiles,
            model_name=model_name,
            work_dir=work_dir,
            seed=seed,
            device_id=device_id,
            **params
        )

    def _build(self, learning_rate: float, batch_size: int, pl_trainer_dict: Dict[str, Any]) -> Any:
        add_relative_index = self.params.get('add_relative_index', True)

        self._model = TFTModel(
            input_chunk_length=self.timeframe_size,
            output_chunk_length=self.prediction_len,
            likelihood=self.loss_fn,
            optimizer_kwargs={'lr': learning_rate},
            batch_size=batch_size,
            model_name=self.model_name,
            work_dir=self.work_dir,
            save_checkpoints=True,
            pl_trainer_kwargs=pl_trainer_dict,
            add_relative_index=add_relative_index,
            random_state=self.seed
        )
