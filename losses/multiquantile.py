import torch


class MultiQuantileLoss(torch.nn.Module):
    def __init__(self, quantiles):
        super(MultiQuantileLoss, self).__init__()

        self.quantiles = quantiles

    def forward(self, y_pred, y_true, quantiles: list[float] or None = None):
        if quantiles is None:
            quantiles = self.quantiles

        losses = []

        for i, q in enumerate(quantiles):
            errors = y_true - y_pred[:, i]
            losses.append(torch.max((q - 1)*errors, q*errors).unsqueeze(1))
        loss = torch.mean(torch.sum(torch.cat(losses, dim=1), dim=1))
        return loss
