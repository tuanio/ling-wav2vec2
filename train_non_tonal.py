from mdd.datasets import SupervisedDataset, collate_fn
from mdd.models import (
    EffConformer,
    Conformer,
    ConvLSTMNet,
    ConformerPitch,
    ConformerPitchTonal,
)
from mdd.utils import greedy_decode, decode_phone
import lightning.pytorch as pl
from torch.utils.data import DataLoader
import torch
from torch import nn, optim
import os
import jiwer


class CTCModel(pl.LightningModule):
    def __init__(
        self,
        input_dim=80,
        model_dim=144,
        num_heads=4,
        num_layers=[2, 2, 4],
        model_kernel_size=15,
        dropout=0.2,
        vocab_size=123,
        num_tonals=6,
        lr=3e-4,
        cfg_lr_scheduler=None,
    ):
        super().__init__()
        # self.model = EffConformer(
        #     input_dim=input_dim,
        #     model_dim=model_dim,
        #     num_heads=num_heads,
        #     num_layers=num_layers,
        #     model_kernel_size=model_kernel_size,
        #     vocab_size=vocab_size
        # )
        # self.model = Conformer(
        #     input_dim=input_dim,
        #     model_dim=model_dim,
        #     num_heads=num_heads,
        #     num_layers=num_layers,
        #     model_kernel_size=model_kernel_size,
        #     dropout=dropout,
        #     vocab_size=vocab_size
        # )

        self.model = ConformerPitch(
            input_dim=input_dim,
            model_dim=model_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            model_kernel_size=model_kernel_size,
            dropout=dropout,
            vocab_size=vocab_size,
        )

        # self.model = ConvLSTMNet(input_dim, model_dim,
        #                 num_layers=num_layers, vocab_size=vocab_size)
        self.lr = lr
        self.cfg_lr_scheduler = cfg_lr_scheduler
        self.cfg_lr_scheduler["max_lr"] = lr
        self.criterion = nn.CTCLoss()

        self.actual_phonemes = []
        self.predict_phonemes = []

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        lr_scheduler = {
            "scheduler": optim.lr_scheduler.OneCycleLR(
                optimizer, **self.cfg_lr_scheduler
            ),
            "name": "lr_scheduler_logger",
            "interval": "step",  # or 'epoch'
            "frequency": 1,
        }
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def training_step(self, batch, batch_idx):
        x, x_len, y, y_len, pitches, tonal, tonal_len = batch

        log_probs, out_len = self.model(
            x.permute(0, 2, 1), x_len, pitches.permute(0, 2, 1)
        )

        # after: log_probs (bs, seq len, vocab size)

        loss = self.criterion(log_probs.permute(1, 0, 2), y, out_len, y_len)

        self.log("train_loss", loss, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, x_len, y, y_len, pitches, tonal, tonal_len = batch

        log_probs, out_len = self.model(
            x.permute(0, 2, 1), x_len, pitches.permute(0, 2, 1)
        )

        # after: log_probs (bs, seq len, vocab size)

        loss = self.criterion(log_probs.permute(1, 0, 2), y, out_len, y_len)

        self.log("val_loss", loss, sync_dist=True)

        actuals = [decode_phone(i.detach().cpu().tolist()) for i in y]
        predicts = greedy_decode(log_probs.argmax(dim=-1).detach().cpu())

        self.actual_phonemes.extend(actuals)
        self.predict_phonemes.extend(predicts)

    def on_validation_epoch_end(self):
        all_actuals = self.actual_phonemes
        all_predicts = self.predict_phonemes

        all_actuals = [" ".join(i) for i in all_actuals]
        all_predicts = [" ".join(i) for i in all_predicts]

        wer = jiwer.wer(all_actuals, all_predicts)

        self.log("val_wer", wer, sync_dist=True)

        self.actual_phonemes.clear()
        self.predict_phonemes.clear()

    def test_step(self, batch, batch_idx):
        x, x_len, y, y_len, pitches, tonal, tonal_len = batch

        log_probs, out_len = self.model(
            x.permute(0, 2, 1), x_len, pitches.permute(0, 2, 1)
        )

        # after: log_probs (bs, seq len, vocab size)

        loss = self.criterion(log_probs.permute(1, 0, 2), y, out_len, y_len)

        self.log("test_loss", loss, sync_dist=True)

        self.log("test_loss", loss, sync_dist=True)

        actuals = [decode_phone(i.detach().cpu().tolist()) for i in y]
        predicts = greedy_decode(log_probs.argmax(dim=-1).detach().cpu())

        self.actual_phonemes.extend(actuals)
        self.predict_phonemes.extend(predicts)

    def on_test_epoch_end(self):
        all_actuals = self.actual_phonemes
        all_predicts = self.predict_phonemes

        all_actuals = [" ".join(i) for i in all_actuals]
        all_predicts = [" ".join(i) for i in all_predicts]

        wer = jiwer.wer(all_actuals, all_predicts)

        self.log("test_wer", wer, sync_dist=True)

        self.actual_phonemes.clear()
        self.predict_phonemes.clear()


data_path = "/data/tuanio/data/share_with_150/data_vlsp_md_d_2023/splitted_data_113"

train_dataset = SupervisedDataset(os.path.join(data_path, "train.json"), True)
test_dataset = SupervisedDataset(os.path.join(data_path, "test.json"))

print("train: {}, test: {}".format(len(train_dataset), len(test_dataset)))

batch_size = 16
num_workers = 4
lr = 3e-4
max_epochs = 700
total_steps = len(train_dataset) * max_epochs

train_loader = DataLoader(
    train_dataset,
    batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
    collate_fn=collate_fn,
)
test_loader = DataLoader(
    test_dataset,
    batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
    collate_fn=collate_fn,
)

input_dim = 81
model_dim = 128
num_heads = 4
num_layers = 4
dropout = 0.2
model_kernel_size = 31
vocab_size = 123

cfg_lr_scheduler = {"pct_start": 0.1, "total_steps": total_steps // batch_size}

# model = CTCModel(
#     input_dim=80,
#     model_dim=96,
#     num_heads=4,
#     num_layers=[2, 2, 4],
#     model_kernel_size=15,
#     vocab_size=123,
#     lr=lr,
#     cfg_lr_scheduler=cfg_lr_scheduler
# )

model = CTCModel(
    input_dim=input_dim,
    model_dim=model_dim,
    num_heads=num_heads,
    num_layers=num_layers,
    dropout=dropout,
    model_kernel_size=model_kernel_size,
    vocab_size=vocab_size,
    lr=lr,
    cfg_lr_scheduler=cfg_lr_scheduler,
)

# model = CTCModel(
#     input_dim=80,
#     model_dim=64,
#     num_layers=2,
#     vocab_size=123,
#     lr=lr,
#     cfg_lr_scheduler=cfg_lr_scheduler
# )

print(model)

print("Number of params:", sum(p.numel() for p in model.parameters()))

# wandb_logger = None
#
name = f"conformer_dim{model_dim}_heads{num_heads}_layers{num_layers}_drop{dropout}_kernel{model_kernel_size}_nfft512_hop128_warmup0.2_{max_epochs}epochs_inputdim81_pitch"
wandb_logger = pl.loggers.WandbLogger(project="md_d_vlsp_2023", name=name)

trainer = pl.Trainer(
    devices=-1,
    accelerator="gpu",
    precision=16,
    max_epochs=max_epochs,
    logger=wandb_logger,
    log_every_n_steps=50,
)

trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=test_loader)
