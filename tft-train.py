import os
import pickle
import numpy as np
import pandas as pd
import torch
from data import utils
from data.loader import DatasetLoader
from models.darts.model import DartsModel
from models.darts.tft import TFT

dataset_directory = 'data/datasets'
large_configs_directory = 'experiments/configs/datasets/large'
small_configs_directory = 'experiments/configs/datasets/small'
large_work_dir = 'experiments/checkpoints/tft'
large_model_name = 'large'
large_tensorboard_dir = 'experiments/tensorboard/tft/large'
small_finetuned_work_dir = 'experiments/checkpoints/tft'
small_finetuned_model_name = 'small-finetuned'
small_finetuned_tensorboard_dir = 'experiments/tensorboard/tft/small-finetuned'
small_work_dir = 'experiments/checkpoints/tft'
small_model_name = 'small'
small_tensorboard_dir = 'experiments/tensorboard/tft/small'
device_id = 0
seed = 0

timeframe_size = 28
noise_percentage = 0.0
quantiles = [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
lr = 0.001
finetuning_lr = 0.0005
batch_size = 64
epochs = 200
early_stopping_patience = 50
params = {'add_relative_index': True}


def train(
        train_df: pd.DataFrame,
        eval_df: pd.DataFrame,
        test_df: pd.DataFrame,
        device_id: int,
        learning_rate: float,
        model_name: str,
        work_dir: str,
        tensorboard_dir: str,
        apply_transfer_learning: bool = False
) -> (DartsModel, pd.DataFrame):
    def train_model() -> DartsModel:
        print('Converting timeseries to TimeSeries datasets...')

        x_train, y_train, _ = utils.construct_timeseries_dataset(df=train_df, timeframe_size=timeframe_size, scale_data=True)
        x_eval, y_eval, _ = utils.construct_timeseries_dataset(df=eval_df, timeframe_size=timeframe_size, scale_data=True)

        print(f'Training Darts Model')

        model = TFT(
            timeframe_size=timeframe_size,
            prediction_len=1,
            quantiles=quantiles,
            model_name=model_name,
            work_dir=work_dir,
            seed=seed,
            device_id=device_id,
            ** params
        )
        model.build(
            learning_rate=learning_rate,
            batch_size=batch_size,
            early_stopping_patience=early_stopping_patience,
            tensorboard_dir=tensorboard_dir
        )

        if apply_transfer_learning:
            print('\n--- Transfer Learning from Large Model ---\n')

            model.load_checkpoint(model_name=large_model_name, work_dir=large_work_dir)

        model.train_model(
            x_train=x_train,
            y_train=y_train,
            x_eval=x_eval,
            y_eval=y_eval,
            epochs=epochs
        )
        return model

    def evaluate_model(model: DartsModel) -> pd.DataFrame:
        evaluation_dict = {'exchange': [], 'symbol': [], 'mse': [], 'mae': []}
        for _, df_group in test_df.groupby(['exchange', 'symbol']):
            exchange = df_group.iloc[0]['exchange']
            symbol = df_group.iloc[0]['symbol']

            x_test, y_test, _ = utils.construct_timeseries_dataset(df=df_group, timeframe_size=timeframe_size, scale_data=True)
            mse, mae = model.eval_model(x_test=x_test, y_test=y_test)

            print(f'Evaluated {exchange}-{symbol} with mse={mse}, mae={mae}')

            evaluation_dict['exchange'].append(exchange)
            evaluation_dict['symbol'].append(symbol)
            evaluation_dict['mse'].append(mse)
            evaluation_dict['mae'].append(mae)
        return pd.DataFrame(evaluation_dict)

    model = train_model()
    evaluation_df = evaluate_model(model=model)
    return model, evaluation_df


def predict(
        test_df: pd.DataFrame,
        device_id: int,
        learning_rate: float,
        model_name: str,
        work_dir: str,
        tensorboard_dir: str
) -> (DartsModel, pd.DataFrame):
    os.makedirs(name='experiments/predictions/checkpoints/', exist_ok=True)
    model = TFT(
        timeframe_size=timeframe_size,
        prediction_len=1,
        quantiles=quantiles,
        model_name=model_name,
        work_dir='experiments/predictions/checkpoints/',
        seed=seed,
        device_id=device_id,
        **params
    )
    model.build(
        learning_rate=learning_rate,
        batch_size=batch_size,
        early_stopping_patience=early_stopping_patience,
        tensorboard_dir=tensorboard_dir
    )
    model.load_checkpoint(model_name=model_name, work_dir=work_dir)

    prediction_dict = {'exchange': [], 'symbol': [], 'pred_price': [], 'actual_price': []}
    for _, df_group in test_df.groupby(['exchange', 'symbol']):
        exchange = df_group.iloc[0]['exchange']
        symbol = df_group.iloc[0]['symbol']

        close_prices = df_group.iloc[timeframe_size - 1: -1]['close'].to_numpy()
        actual_prices = df_group.iloc[timeframe_size:]['close'].to_numpy()

        x_test, y_test, scaler = utils.construct_timeseries_dataset(df=df_group, timeframe_size=timeframe_size, scale_data=True)
        y_pred_scaled = model.predict(x=x_test, y=y_test).values().flatten()

        assert df_group.columns[-1] == 'targets'

        dummy_array = np.zeros(shape=(df_group.shape[0] - timeframe_size, x_test[0].values().shape[1]), dtype=np.float64)
        dummy_array[:, -1] = y_pred_scaled
        dummy_array = scaler.inverse_transform(dummy_array)
        y_pred = dummy_array[:, -1]

        pred_prices = (close_prices*np.exp(y_pred))


        prediction_dict['exchange'].extend([exchange]*y_pred.shape[0])
        prediction_dict['symbol'].extend([symbol]*y_pred.shape[0])
        prediction_dict['pred_price'].extend(pred_prices.tolist())
        prediction_dict['actual_price'].extend(actual_prices.tolist())
    return pd.DataFrame(prediction_dict)


def main():
    device_id = 1 if torch.cuda.is_available() else -1
    dataset_loader = DatasetLoader(dataset_directory=dataset_directory)

    # Loading large dataset (train-eval-test)
    with open(f'{large_configs_directory}/train.pkl', 'rb') as dictfile:
        train_df = dataset_loader.load_datasets(dataset_configs=pickle.load(dictfile), noise_percentage=noise_percentage)
    with open(f'{large_configs_directory}/eval.pkl', 'rb') as dictfile:
        eval_df = dataset_loader.load_datasets(dataset_configs=pickle.load(dictfile))
    with open(f'{large_configs_directory}/test.pkl', 'rb') as dictfile:
        test_df = dataset_loader.load_datasets(dataset_configs=pickle.load(dictfile))

    print(
        f'Loaded {train_df.shape[0]} train samples, {eval_df.shape[0]} eval samples and {test_df.shape[0]} test samples from large dataset.')

    # Train TFT-Large model
    print('\n#--- Training TFT-Large ---\n')

    _, evaluation_df = train(
        train_df=train_df,
        eval_df=eval_df,
        test_df=test_df,
        device_id=device_id,
        learning_rate=lr,
        model_name=large_model_name,
        work_dir=large_work_dir,
        tensorboard_dir=large_tensorboard_dir,
        apply_transfer_learning=False
    )
    evaluation_df.to_csv('experiments/results/tft/large_evaluation.csv', index=False)

    print('\n#--- Predicting Time Series using TFT-Large ---\n')

    predictions_df = predict(
        test_df=test_df,
        device_id=device_id,
        learning_rate=lr,
        model_name=large_model_name,
        work_dir=large_work_dir,
        tensorboard_dir=large_tensorboard_dir
    )
    predictions_df.to_csv('experiments/results/tft/large_predictions.csv', index=False)

    # Load Small Dataset
    with open(f'{small_configs_directory}/train.pkl', 'rb') as dictfile:
        train_df = dataset_loader.load_datasets(dataset_configs=pickle.load(dictfile), noise_percentage=noise_percentage)
    with open(f'{small_configs_directory}/eval.pkl', 'rb') as dictfile:
        eval_df = dataset_loader.load_datasets(dataset_configs=pickle.load(dictfile))

    # Train TFT-Small-Finetuned
    print('\n#--- Training TFT-Small-Finetuned ---\n')

    _, evaluation_df = train(
        train_df=train_df,
        eval_df=eval_df,
        test_df=test_df,
        device_id=device_id,
        learning_rate=finetuning_lr,
        model_name=small_finetuned_model_name,
        work_dir=small_finetuned_work_dir,
        tensorboard_dir=small_finetuned_tensorboard_dir,
        apply_transfer_learning=True
    )
    evaluation_df.to_csv('experiments/results/tft/small_finetuned_evaluation.csv', index=False)

    print('\n#--- Predicting Time Series using TFT-Small-Finetuned ---\n')

    predictions_df = predict(
        test_df=test_df,
        device_id=device_id,
        learning_rate=finetuning_lr,
        model_name=small_finetuned_model_name,
        work_dir=small_finetuned_work_dir,
        tensorboard_dir=small_finetuned_tensorboard_dir
    )
    predictions_df.to_csv('experiments/results/tft/small_finetuned_predictions.csv', index=False)

    print('\n#--- Training TFT-Small-No-Finetuned ---\n')

    _, evaluation_df = train(
        train_df=train_df,
        eval_df=eval_df,
        test_df=test_df,
        device_id=device_id,
        learning_rate=lr,
        model_name=small_model_name,
        work_dir=small_work_dir,
        tensorboard_dir=small_tensorboard_dir,
        apply_transfer_learning=False
    )
    evaluation_df.to_csv('experiments/results/tft/small_evaluation.csv', index=False)

    print('\n#--- Predicting Time Series using TFT-Small-No-Finetuned ---\n')

    predictions_df = predict(
        test_df=test_df,
        device_id=device_id,
        learning_rate=lr,
        model_name=small_model_name,
        work_dir=small_work_dir,
        tensorboard_dir=small_tensorboard_dir
    )
    predictions_df.to_csv('experiments/results/tft/small_predictions.csv', index=False)


if __name__ == "__main__":
    if not 0 <= device_id <= 1:
        raise RuntimeError(f'Maximum 2 GPUs are supported with ids 0 or 1, got {device_id}')

    os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)

    main()
