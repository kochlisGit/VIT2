import torch


class RegressionMetrics:
    def __init__(self, use_cuda: bool):
        self._use_cuda = use_cuda

    def mse(self, targets: torch.Tensor, predictions: torch.Tensor):
        error = torch.square(predictions - targets).mean()
        return error.detach().cpu().numpy() if self._use_cuda else error

    def mae(self, targets: torch.Tensor, predictions: torch.Tensor):
        error = torch.abs(predictions - targets).mean()
        return error.detach().cpu().item() if self._use_cuda else error
