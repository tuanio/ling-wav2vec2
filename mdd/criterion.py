import torch
from torch import nn


class FocalCTCLoss(nn.Module):
    """
    reference: https://www.hindawi.com/journals/complexity/2019/9345861/
    """

    def __init__(self, alpha, gamma, blank=0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ctc_crit = nn.CTCLoss(blank=blank, reduction="none")

    def forward(self, x, y, x_len, y_len):
        a = self.alpha
        g = self.gamma

        ctc_loss = self.ctc_crit(x, y, x_len, y_len)
        p = torch.exp(-ctc_loss)
        focal_loss = a * ((1 - p) ** g) * ctc_loss
        return focal_loss.mean()
