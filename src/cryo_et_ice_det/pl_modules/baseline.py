import os
import ast
import random
from argparse import ArgumentParser, Namespace

import cv2
import numpy as np
import pandas as pd
import torchmetrics
import torch
import torch.nn.functional as F
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
import torchvision.transforms as transforms
from lightning.pytorch import LightningModule, Trainer, cli_lightning_logo
from PIL import Image
import segmentation_models_pytorch as smp 
from torch import nn
from torch.utils.data import DataLoader, Dataset
from IPython import embed
from dotenv import load_dotenv

from nn_core.common import PROJECT_ROOT

load_dotenv()


class SegModel(LightningModule):
  def __init__(
    self,
    batch_size: int = 4,
    lr: float = 1e-3,
    *args,
    **kwargs,
  ):
    super().__init__()
    self.save_hyperparameters()
    self.batch_size = batch_size
    self.lr = lr

    self.net = smp.FPN(
    encoder_name='resnet34',
    encoder_weights=None,
    in_channels=1,
    classes=1
    )
    self.transform = None
    self.metric_f1_fn = torchmetrics.functional.classification.binary_f1_score
    self.metric_iou_fn = torchmetrics.functional.classification.binary_jaccard_index

  def forward(self, x):
    return self.net(x)

  def training_step(self, batch, batch_idx):
    img, mask = batch
    img = img.float()
    out = self(img)
    loss = F.binary_cross_entropy_with_logits(out, mask.float())
    f1 = self.metric_f1_fn(out, mask)
    iou = self.metric_iou_fn(out, mask)
    self.log('loss/train', loss, prog_bar=True, on_step=True, on_epoch=True)
    self.log('iou/train', iou, prog_bar=False, on_step=True, on_epoch=True)
    self.log('f1/train', f1, prog_bar=False, on_step=True, on_epoch=True)

    return loss

  def validation_step(self, batch, batch_idx):
    img, mask = batch
    img = img.float()
    out = self(img)
    loss = F.binary_cross_entropy_with_logits(out, mask.float())
    f1 = self.metric_f1_fn(out, mask)
    iou = self.metric_iou_fn(out, mask)
    self.log('loss/val', loss, prog_bar=True, on_step=True, on_epoch=True)
    self.log('iou/val', iou, prog_bar=False, on_step=True, on_epoch=True)
    self.log('f1/val', f1, prog_bar=False, on_step=True, on_epoch=True)

  def configure_optimizers(self):
    opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
    # sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
    return [opt]#, [sch]

  def train_dataloader(self):
    return DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True, num_workers=os.cpu_count())

  def val_dataloader(self):
    return DataLoader(self.validset, batch_size=self.batch_size, shuffle=False, num_workers=os.cpu_count())

def main(hparams: Namespace):
  # ------------------------
  # 1 INIT LIGHTNING MODEL
  # ------------------------
  model = SegModel(**vars(hparams))

  # ------------------------
  # 2 INIT TRAINER
  # ------------------------

  logger = TensorBoardLogger(
    save_dir=PROJECT_ROOT / "baseline_logs",
    name=hparams.exp_name
  )
  ckpt_dirpath = os.path.join(logger.log_dir, 'ckpts')
  checkpoint_callback = ModelCheckpoint(
    dirpath=ckpt_dirpath,
    save_top_k=10,
    monitor="iou/val",
    mode='max',
    filename="epoch={epoch:02d}-step={step}",
    save_last=True,
    auto_insert_metric_name=True
  )
  lr_logger = LearningRateMonitor(logging_interval='step')
  trainer = Trainer(
    devices=1,
    logger=logger,
    precision=hparams.precision,
    callbacks=[lr_logger, checkpoint_callback],
    num_sanity_val_steps=0,
    log_every_n_steps=10,
    check_val_every_n_epoch=5
  )

  # ------------------------
  # 3 START TRAINING
  # ------------------------
  trainer.fit(model)


if __name__ == "__main__":
    cli_lightning_logo()

    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, help="path where dataset is stored")
    parser.add_argument("--batch_size", type=int, default=20, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.001, help="adam: learning rate")
    parser.add_argument("--exp_name", type=str, default='test', help="name of the experiment")
    parser.add_argument("--precision", type=str, default='16', help="precision to use during training")
    hparams = parser.parse_args()

    main(hparams)