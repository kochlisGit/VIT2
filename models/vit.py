import os
import numpy as np
import torch
import timm
from torchview import draw_graph
from tqdm import tqdm
from data import utils
from losses.multiquantile import MultiQuantileLoss
from metrics.regression import RegressionMetrics


class ViTRegressor(torch.nn.Module):
    def __init__(self, num_channels: int, image_size: tuple[int, int], quantiles: list[float], device: torch.device = 'cpu'):
        super(ViTRegressor, self).__init__()

        assert 0.5 in quantiles, f't=0.5 is required for multi-quantile loss, got {quantiles}'

        self._num_channels = num_channels
        self._image_size = image_size

        use_cuda = torch.cuda.is_available() and device != 'cpu'
        self.device = device if use_cuda else torch.device('cpu')

        model = timm.create_model(
            model_name='vit_base_patch16_224',
            in_chans=num_channels,
            img_size=image_size,
            num_classes=0,
            pretrained=False
        )
        head = torch.nn.Linear(model.num_features, len(quantiles))
        self.loss = MultiQuantileLoss(quantiles=quantiles)
        self._metrics = RegressionMetrics(use_cuda=use_cuda)
        self._num_quantiles = len(quantiles)
        self._quantile_05_index = quantiles.index(0.5)

        self.model = torch.nn.Sequential(model, head)

        if use_cuda:
            self.model = self.model.to(device=self.device)

        self.train_loss_per_epoch = []
        self.eval_loss_per_epoch = []
        self.eval_mse_per_epoch = []
        self.eval_mae_per_epoch = []

    def display(self, save_graph: bool = False, summary: bool = True):
        if save_graph:
            try:
                draw_graph(
                    self.model,
                    input_size=(1, self._num_channels, self._image_size[0], self._image_size[1]),
                    device=self.device,
                    expand_nested=True,
                    graph_name='Vit',
                    save_graph=True,
                    filename='vit.png'
                )
            except Exception as e:
                print(e)

        if summary:
            print(self.model)

    def save(self, checkpoint_directory: str):
        dir_path = checkpoint_directory.rsplit('/', maxsplit=1)[0]
        os.makedirs(name=dir_path, exist_ok=True)

        torch.save(self.model.state_dict(), checkpoint_directory)

    def load(self, checkpoint_directory: str):
        self.model.load_state_dict(torch.load(checkpoint_directory))

    def forward(self, x):
        return self.model(x)

    def eval_model(self, eval_dataloader: torch.utils.data.DataLoader, desc_prefix: str or None = None) -> (float, float, float):
        if desc_prefix is None:
            desc_prefix = 'Evaluation'

        total_loss = 0.0
        total_mse = 0.0
        total_mae = 0.0
        num_batches = len(eval_dataloader)

        self.model.eval()
        with tqdm(total=num_batches, desc=desc_prefix) as pbar:
            for batch in eval_dataloader:
                with torch.no_grad():
                    inputs, targets = batch
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)

                    outputs = self.forward(x=inputs)
                    total_loss += self.loss(y_pred=outputs, y_true=targets).item()
                    total_mae += self._metrics.mae(targets=targets, predictions=outputs[:, self._quantile_05_index])
                    total_mse += self._metrics.mse(targets=targets, predictions=outputs[:, self._quantile_05_index])

                    pbar.update(1)
                    pbar.set_description(desc=f'{desc_prefix} - Val Loss: {total_loss:.4f}, Val MSE: {total_mse:.4f}, Val MAE: {total_mae:.4f}')

            total_loss/=num_batches
            total_mse/=num_batches
            total_mae/=num_batches
            pbar.set_description(desc=f'{desc_prefix} - Val Loss: {total_loss:.4f}, Val MSE: {total_mse:.4f}, Val MAE: {total_mae:.4f}')
        return total_loss, total_mse, total_mae

    def train_model(
            self,
            x_train: np.ndarray,
            y_train: np.ndarray,
            x_eval: np.ndarray,
            y_eval: np.ndarray,
            lr: float,
            batch_size: int,
            epochs: int,
            early_stopping_patience: int,
            autosave: bool,
            checkpoint_directory: str
    ) -> (list[float], list[float], list[float], list[float]):
        train_dataloader = utils.construct_dataloader(x=x_train, y=y_train, batch_size=batch_size, shuffle=True)
        eval_dataloader = utils.construct_dataloader(x=x_eval, y=y_eval, batch_size=1024, shuffle=False)

        self.train_loss_per_epoch = []
        self.eval_loss_per_epoch = []
        self.eval_mse_per_epoch = []
        self.eval_mae_per_epoch = []

        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=lr)
        last_early_stopping_epoch = 0
        best_val_loss = np.inf
        num_batches = len(train_dataloader)

        for epoch in range(epochs):
            self.model.train()

            total_loss = 0.0
            with tqdm(total=num_batches) as pbar:
                pbar.set_description(desc=f'Training: Epoch: {epoch}/{epochs}')

                for i, (inputs, targets) in enumerate(train_dataloader):
                    optimizer.zero_grad()
                    outputs = self.forward(x=inputs.to(self.device))
                    loss = self.loss.forward(y_pred=outputs, y_true=targets.to(self.device))
                    loss.backward()
                    optimizer.step()

                    batch_loss = loss.item()
                    total_loss += batch_loss

                    pbar.set_description(desc=f'Epoch: {epoch + 1}/{epochs} - Batch: {i + 1}/{num_batches}, Batch Loss = {batch_loss}')
                    pbar.update(1)
                total_loss/=num_batches
                pbar.set_description(desc=f'Epoch: {epoch + 1}/{epochs}, Total Train Loss = {total_loss:.4f}')
            self.train_loss_per_epoch.append(total_loss)

            eval_loss, eval_mse, eval_mae = self.eval_model(eval_dataloader=eval_dataloader, desc_prefix=f'Epoch: {epoch}/{epochs}')
            self.eval_loss_per_epoch.append(eval_loss)
            self.eval_mse_per_epoch.append(eval_mse)
            self.eval_mae_per_epoch.append(eval_mae)

            if eval_loss < best_val_loss:
                best_val_loss = eval_loss
                last_early_stopping_epoch = epoch

                if autosave:
                    self.save(checkpoint_directory=checkpoint_directory)

                print(f'\nFound new best loss: {best_val_loss} at epoch {epoch}')
            if epoch - last_early_stopping_epoch > early_stopping_patience:
                print(f'\nNo model improvement has been achieved in {early_stopping_patience} epochs, since {last_early_stopping_epoch}. Early Stopping.')

                break
        return self.train_loss_per_epoch, self.eval_loss_per_epoch, self.eval_mse_per_epoch, self.eval_mae_per_epoch
