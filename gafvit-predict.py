import os
import pickle
import numpy as np
import pandas as pd
import torch
from typing import Optional
from data import utils
from data.loader import DatasetLoader
from models.vit import ViTRegressor

dataset_directory = 'data/datasets'
large_checkpoint_directory = 'experiments/checkpoints/gaf-vit/large/state_dict_model.pt'
small_checkpoint_directory_finetuned = 'experiments/checkpoints/gaf-vit/small-finetuned/state_dict_model.pt'
small_checkpoint_directory = 'experiments/checkpoints/gaf-vit/small/state_dict_model.pt'
large_configs_directory = 'experiments/configs/datasets/large'
small_configs_directory = 'experiments/configs/datasets/small'
device_id = 0

timeframe_size = 28
noise_percentage = 0.0
quantiles = [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
lr = 0.001
finetuning_lr = 0.0005
batch_size = 64
epochs = 600
early_stopping_patience = 200
autosave = True


def predict(
        test_df: pd.DataFrame,
        device: torch.device,
        previous_checkpoint_directory: Optional[str] = None
) -> pd.DataFrame:
    prediction_dict = {'exchange': [], 'symbol': [], 'pred_price': [], 'actual_price': []}
    for _, df_group in test_df.groupby(['exchange', 'symbol']):
        exchange = df_group.iloc[0]['exchange']
        symbol = df_group.iloc[0]['symbol']
        x_test, _ = utils.construct_image_dataset(df=df_group, timeframe_size=timeframe_size)
        model = ViTRegressor(
            num_channels=x_test.shape[1],
            image_size=(timeframe_size, timeframe_size),
            quantiles=quantiles,
            device=device
        )
        model.load(previous_checkpoint_directory)

        close_prices = df_group.iloc[timeframe_size - 1: -1]['close'].to_numpy()
        actual_prices = df_group.iloc[timeframe_size:]['close'].to_numpy()
        x_test = torch.tensor(data=x_test, dtype=torch.float32, device=device)
        y_pred_quantiles = model.forward(x=x_test).detach().cpu().numpy()

        y_pred = y_pred_quantiles[:, quantiles.index(0.5)]
        pred_prices = (close_prices * np.exp(y_pred))

        prediction_dict['exchange'].extend([exchange] * y_pred.shape[0])
        prediction_dict['symbol'].extend([symbol] * y_pred.shape[0])
        prediction_dict['pred_price'].extend(pred_prices.tolist())
        prediction_dict['actual_price'].extend(actual_prices.tolist())
    return pd.DataFrame(prediction_dict)


def main():
    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    dataset_loader = DatasetLoader(dataset_directory=dataset_directory)

    # Loading large dataset (train-eval-test)
    with open(f'{large_configs_directory}/train.pkl', 'rb') as dictfile:
        train_df = dataset_loader.load_datasets(dataset_configs=pickle.load(dictfile), noise_percentage=noise_percentage)
    with open(f'{large_configs_directory}/eval.pkl', 'rb') as dictfile:
        eval_df = dataset_loader.load_datasets(dataset_configs=pickle.load(dictfile))
    with open(f'{large_configs_directory}/test.pkl', 'rb') as dictfile:
        test_df = dataset_loader.load_datasets(dataset_configs=pickle.load(dictfile))

    print(f'Loaded {train_df.shape[0]} train samples, {eval_df.shape[0]} eval samples and {test_df.shape[0]} test samples from large dataset.')

    # Predict ViT-Large model
    print('\n#--- Predicting Time Series using NHits-Large ---\n')

    predictions_df = predict(
        test_df=test_df,
        device=device,
        previous_checkpoint_directory=large_checkpoint_directory
    )
    predictions_df.to_csv('experiments/results/gaf-vit/large_predictions.csv', index=False)

   # Load Small Dataset
    with open(f'{small_configs_directory}/train.pkl', 'rb') as dictfile:
        train_df = dataset_loader.load_datasets(dataset_configs=pickle.load(dictfile), noise_percentage=noise_percentage)
    with open(f'{small_configs_directory}/eval.pkl', 'rb') as dictfile:
        eval_df = dataset_loader.load_datasets(dataset_configs=pickle.load(dictfile))

    # Predict ViT-Small-Finetuned
    print('\n#--- Predicting Time Series using NHits-Small-Finetuned ---\n')

    predictions_df = predict(
        test_df=test_df,
        device=device,
        previous_checkpoint_directory=small_checkpoint_directory_finetuned
    )
    predictions_df.to_csv('experiments/results/gaf-vit/small_finetuned_predictions.csv', index=False)

    print('\n#--- Predict ViT-Small-No-Finetuned ---\n')

    predictions_df = predict(
        test_df=test_df,
        device=device,
        previous_checkpoint_directory=small_checkpoint_directory_finetuned
    )
    predictions_df.to_csv('experiments/results/gaf-vit/small_predictions.csv', index=False)


if __name__ == "__main__":
    if not 0 <= device_id <= 1:
        raise RuntimeError(f'Maximum 2 GPUs are supported with ids 0 or 1, got {device_id}')

    os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)

    main()
