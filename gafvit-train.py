import os
import pickle
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


def train(
        train_df: pd.DataFrame,
        eval_df: pd.DataFrame,
        test_df: pd.DataFrame,
        device: torch.device,
        learning_rate: float,
        checkpoint_directory: str,
        previous_checkpoint_directory: Optional[str] = None
) -> (ViTRegressor, pd.DataFrame, pd.DataFrame):
    def train_model() -> (ViTRegressor, pd.DataFrame):
        print('Converting timeseries to images...')

        x_train, y_train = utils.construct_image_dataset(df=train_df, timeframe_size=timeframe_size)
        x_eval, y_eval = utils.construct_image_dataset(df=eval_df, timeframe_size=timeframe_size)

        print(f'Constructed {x_train.shape[0]} train and {x_eval.shape} eval images of size: {x_train.shape[1:]}')

        model = ViTRegressor(
            num_channels=x_train.shape[1],
            image_size=(timeframe_size, timeframe_size),
            quantiles=quantiles,
            device=device
        )

        if previous_checkpoint_directory is not None:
            model.load(previous_checkpoint_directory)

        train_loss_per_epoch, eval_loss_per_epoch, eval_mse_per_epoch, eval_mae_per_epoch = model.train_model(
            x_train=x_train,
            y_train=y_train,
            x_eval=x_eval,
            y_eval=y_eval,
            lr=learning_rate,
            batch_size=batch_size,
            epochs=epochs,
            early_stopping_patience=early_stopping_patience,
            autosave=True,
            checkpoint_directory=checkpoint_directory
        )
        history_df = pd.DataFrame({
            'epochs': range(1, len(train_loss_per_epoch) + 1),
            'train-loss': train_loss_per_epoch,
            'eval-loss': eval_loss_per_epoch,
            'eval-mse': eval_mae_per_epoch,
            'eval_mae': eval_mse_per_epoch
        })
        return model, history_df

    def evaluate_model(model: ViTRegressor) -> pd.DataFrame:
        evaluation_dict = {'exchange': [], 'symbol': [], 'loss': [], 'mse': [], 'mae': []}
        for _, df_group in test_df.groupby(['exchange', 'symbol']):
            exchange = df_group.iloc[0]['exchange']
            symbol = df_group.iloc[0]['symbol']

            x_test, y_test = utils.construct_image_dataset(df=df_group, timeframe_size=timeframe_size)
            test_dataloader = utils.construct_dataloader(x=x_test, y=y_test, batch_size=batch_size, shuffle=False)
            loss, mse, mae = model.eval_model(eval_dataloader=test_dataloader, desc_prefix=f'Evaluating: {exchange}-{symbol}')
            evaluation_dict['exchange'].append(exchange)
            evaluation_dict['symbol'].append(symbol)
            evaluation_dict['loss'].append(loss)
            evaluation_dict['mse'].append(mse)
            evaluation_dict['mae'].append(mae)
        return pd.DataFrame(evaluation_dict)


    model, history_df = train_model()
    evaluation_df = evaluate_model(model=model)
    return model, history_df, evaluation_df


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

    # Train ViT-Large model
    print('\n#--- Training ViT-Large ---\n')

    _, history_df, evaluation_df = train(
        train_df=train_df,
        eval_df=eval_df,
        test_df=test_df,
        device=device,
        learning_rate=lr,
        checkpoint_directory=large_checkpoint_directory,
        previous_checkpoint_directory=None
    )
    history_df.to_csv('experiments/results/gaf-vit/large_history.csv', index=False)
    evaluation_df.to_csv('experiments/results/gaf-vit/large_evaluation.csv', index=False)

   # Load Small Dataset
    with open(f'{small_configs_directory}/train.pkl', 'rb') as dictfile:
        train_df = dataset_loader.load_datasets(dataset_configs=pickle.load(dictfile), noise_percentage=noise_percentage)
    with open(f'{small_configs_directory}/eval.pkl', 'rb') as dictfile:
        eval_df = dataset_loader.load_datasets(dataset_configs=pickle.load(dictfile))

    # Train ViT-Small-Finetuned
    print('\n#--- Training ViT-Small-Finetuned ---\n')

    _, history_df, evaluation_df = train(
        train_df=train_df,
        eval_df=eval_df,
        test_df=eval_df,
        device=device,
        learning_rate=finetuning_lr,
        checkpoint_directory=small_checkpoint_directory_finetuned,
        previous_checkpoint_directory=large_checkpoint_directory
    )
    history_df.to_csv('experiments/results/gaf-vit/small_finetuned_history.csv', index=False)
    evaluation_df.to_csv('experiments/results/gaf-vit/small_finetuned_evaluation.csv', index=False)

    print('\n#--- Training ViT-Small-No-Finetuned ---\n')

    _, history_df, evaluation_df = train(
        train_df=train_df,
        eval_df=eval_df,
        test_df=eval_df,
        device=device,
        learning_rate=lr,
        checkpoint_directory=small_checkpoint_directory,
        previous_checkpoint_directory=None
    )
    history_df.to_csv('experiments/results/gaf-vit/small_history.csv', index=False)
    evaluation_df.to_csv('experiments/results/gaf-vit/small_evaluation.csv', index=False)


if __name__ == "__main__":
    if not 0 <= device_id <= 1:
        raise RuntimeError(f'Maximum 2 GPUs are supported with ids 0 or 1, got {device_id}')

    os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)

    main()
