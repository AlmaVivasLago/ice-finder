import json
import numpy as np
import cv2
import ast
import os
import copy
import argparse
from datetime import date
from tqdm import tqdm
import dotenv
import torch
import torch.nn.functional as F
import pandas as pd
import torchmetrics
from torch.utils.data import Dataset
from hydra.utils import instantiate
import segmentation_models_pytorch as smp
from IPython import embed
from omegaconf import DictConfig, ListConfig

import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

import lightning
import lightning.pytorch as pl

import hydra
import omegaconf

from nn_core.common import PROJECT_ROOT
from pl_modules.baseline import SegModel


pl.seed_everything(10); 
torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False

dotenv.load_dotenv()

DATA_ROOT = os.environ.get('SELECTED_DATA_DPATH')
LABELS_ROOT = os.environ.get('SELECTED_LABELS_DPATH')


def _convert_str_to_list(val):
  if pd.isna(val):
    return val
  try:
    return ast.literal_eval(val)
  except Exception as e:
    raise Exception("Error while converting string to list:" + e)

def _load_image(fpath):
  x = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
  x = torch.tensor(x, dtype=torch.float16).unsqueeze(0)
  x = x / 255.
  return 1 - x

def load_data(data_path, labels_dpath):
  train_fpath = os.path.join(labels_dpath, 'train.csv')
  test_fpath = os.path.join(labels_dpath, 'test.csv')
  train_df = pd.read_csv(train_fpath)
  test_df = pd.read_csv(test_fpath)
  train_df['segmentation'] = train_df.segmentation.apply(_convert_str_to_list)
  test_df['segmentation'] = test_df.segmentation.apply(_convert_str_to_list)
  train_df['fpath'] = train_df.file_name.apply(lambda x: os.path.join(DATA_ROOT, x))
  test_df['fpath'] = test_df.file_name.apply(lambda x: os.path.join(DATA_ROOT, x))
  
  train_images, train_masks = [], []
  test_images, test_masks = [], []

  for _, row in train_df.iterrows():
    img = _load_image(row.fpath)
    mask = np.zeros_like(img, dtype=np.uint8).squeeze()
    if row.num_ann > 0:
      polys = []
      for seg in row.segmentation:
        polys.append(np.array(seg).astype(np.int32).reshape(-1, 2))
      cv2.fillPoly(mask, polys, 1)
    mask = torch.from_numpy(mask)
    train_images.append(img)
    train_masks.append(mask)

  for _, row in test_df.iterrows():
    img = _load_image(row.fpath)
    mask = np.zeros_like(img, dtype=np.uint8).squeeze()
    if row.num_ann > 0:
      polys = []
      for seg in row.segmentation:
        polys.append(np.array(seg).astype(np.int32).reshape(-1, 2))
      cv2.fillPoly(mask, polys, 1)
    mask = torch.from_numpy(mask)
    test_images.append(img)
    test_masks.append(mask)

  train_images = torch.stack(train_images).float()
  train_masks = torch.stack(train_masks).float()
  test_images = torch.stack(test_images).float()
  test_masks = torch.stack(test_masks).float()

  return train_images, train_masks, test_images, test_masks

def save_visualized_images(img_tensor, true_mask_tensor, pred_mask_tensor, folder_path, alpha=0.3):
    # Ensure directory exists
    os.makedirs(folder_path, exist_ok=True)
    
    for i in range(len(img_tensor)):
        img = TF.to_pil_image(img_tensor[i].cpu())
        true_mask = true_mask_tensor[i].squeeze().cpu() 
        pred_mask = pred_mask_tensor[i].squeeze().cpu() > 0.5  
        
        img_np = np.array(img.convert('RGB'))
        true_mask_np = np.array(true_mask, dtype=np.uint8) * 255
        pred_mask_np = np.array(pred_mask, dtype=np.uint8) * 255
        
        true_colored_mask = np.zeros_like(img_np)
        true_colored_mask[..., 1] = true_mask_np 
        pred_colored_mask = np.zeros_like(img_np)
        pred_colored_mask[..., 0] = pred_mask_np  
        
        combined_img = img_np.copy()
        combined_img[true_mask_np > 0] = (1-alpha)*combined_img[true_mask_np > 0] + alpha*true_colored_mask[true_mask_np > 0]
        combined_img[pred_mask_np > 0] = (1-alpha)*combined_img[pred_mask_np > 0] + alpha*pred_colored_mask[pred_mask_np > 0]
        
        plt.imsave(os.path.join(folder_path, f'image_with_mask_{i}.png'), combined_img)

class Test:

  def __init__(self, models, device, save_images):
    self.models = models       
    self.device = f'cuda:{device}' if device is not None else 'cpu'

    self.result_dpath = os.path.join(PROJECT_ROOT, 'k-shot-results', date.today().strftime("%d-%m-%Y"))
    os.makedirs(self.result_dpath, exist_ok=True)
    self.save_images = save_images

    self.metric_f1_fn = torchmetrics.functional.classification.binary_f1_score
    self.metric_iou_fn = torchmetrics.functional.classification.binary_jaccard_index
    self.loss_fn = F.binary_cross_entropy_with_logits

  def _fine_tune(self, model, loss_fn, data, lr, num_iters=1):
    X, y = data
    kshot =  X.shape[0]
    X = X.to(self.device)
    y = y.to(self.device)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    im_dir = os.path.join(self.result_dpath, 'baseline', 'finetune')
    os.makedirs(im_dir, exist_ok=True)
    model_weights_dir = os.path.join(im_dir, 'model_weights')
    os.makedirs(model_weights_dir, exist_ok=True)
    
    with tqdm(total=num_iters, desc="Fine-tuning Progress", dynamic_ncols=True) as pbar:
      for _ in range(num_iters):
          with torch.enable_grad():
            out = model(X).squeeze(1)
            loss = loss_fn(out, y)
            pbar.set_postfix({"Loss": f"{loss.item():.6f}"})
          pbar.update(1)
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

    if self.save_images:
      save_visualized_images(X, y, out, im_dir)

    model_weights_path = os.path.join(model_weights_dir, f'model_weights_iter_{kshot}.ckpt')
    torch.save({
                'epoch': num_iters,
                'global_step': num_iters,
                'pytorch-lightning_version': pl.__version__,
                'state_dict': model.state_dict(),
                'optimizer_states': optimizer.state_dict(),
                'hyper_parameters': model.hparams,  
                'lr_schedulers': [], 
                }, model_weights_path
                )

    return model

  def _eval(self, model, loss_fn, data, im_dir):
    X, y = data
    X = X.to(self.device)
    y = y.to(self.device)
    model.eval()
    with torch.no_grad():
        out = model(X).squeeze(1)
    loss = loss_fn(out, y)

    if self.save_images:
      save_visualized_images(X, y, out, im_dir)

    iou = self.metric_iou_fn(out, y)
    f1 = self.metric_f1_fn(out, y)

    return loss.item(), iou.item(), f1.item()

  def run(self, data_dpath, labels_dpath, shots, fine_tune_dict):
    iou_results = {
      m_name: {f"K{s}": [] for s in shots}
      for m_name in list(self.models.keys())
    }
    f1_results = {
      m_name: {f"K{s}": [] for s in shots}
      for m_name in list(self.models.keys())
    }

    train_images, train_masks, test_images, test_masks = load_data(data_dpath, labels_dpath)  
    test_data = (test_images, test_masks)

    for model_name, _model in self.models.items():
      lr = fine_tune_dict[model_name]['lr']
      num_iter = fine_tune_dict[model_name]['num_iter']
      im_dir = os.path.join(self.result_dpath, model_name, 'eval')
      os.makedirs(im_dir, exist_ok=True)

      for K in shots:
        print(f'Fine tunning model for {str(K)} shot.')
        model = copy.deepcopy(_model)
        model = model.to(self.device)

        fine_tune_data = (train_images[:K], train_masks[:K])
        if K > 0: 
          model = self._fine_tune(model, self.loss_fn, fine_tune_data, lr, num_iter)
        _, test_iou, test_f1 = self._eval(model, self.loss_fn, test_data, im_dir)
        iou_results[model_name][f"K{K}"].append(test_iou)
        f1_results[model_name][f"K{K}"].append(test_f1)
        
    if ~ self.save_images:
      with open(os.path.join(self.result_dpath, "test_iou.json"), "w") as fid:
        json.dump(iou_results, fid)
      with open(os.path.join(self.result_dpath, "test_f1.json"), "w") as fid:
        json.dump(f1_results, fid)
    else:
      print(f"Test results saved to: {os.path.relpath(self.result_dpath)}")

    best_f1_performance = {'mean': -np.inf, 'model': '', 'shot': ''}
    for model_name, shots in f1_results.items():
        for shot, vals in shots.items():
            mean_f1 = np.mean(vals)
            if mean_f1 > best_f1_performance['mean']:
                best_f1_performance = {'mean': mean_f1, 'model': model_name, 'shot': shot}

    best_f1_file_path = os.path.join(self.result_dpath, "best_f1_performance.txt")
    with open(best_f1_file_path, "w") as f:
        f.write(f"Best F1 Score: {best_f1_performance['mean']} from {best_f1_performance['model']} at {best_f1_performance['shot']}")  

    print(f"Best model performance saved to: {os.path.abspath(best_f1_file_path)}")

# def run(cfg: DictConfig) -> str:
# # @hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
# # def main(cfg: omegaconf.DictConfig):
# #   run(cfg)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Run k-shot testing.") 
  parser.add_argument('--baseline_ckpt', default = 'logs/baseline-model-v0.ckpt')
  parser.add_argument('--device', type=int, default=None)
  parser.add_argument('--data_dpath', type=str, default=DATA_ROOT)
  parser.add_argument('--labels_dpath', type=str, default=LABELS_ROOT)
  parser.add_argument('--finetune_dict', type=dict, default= {'baseline':dict(lr=0.01, num_iter=30)})
  parser.add_argument('--shots_to_test', nargs='*', type=int, default=[0, 1, 2, 3, 4, 5])
  parser.add_argument('--save_images', action='store_true', help="Flag to save output images with predictions and ground truth labels.")

  args = parser.parse_args()

  #baseline = SegModel(data_path=args.labels_dpath)
  baseline = SegModel.load_from_checkpoint(args.baseline_ckpt)
  #baseline = baseline.load_from_checkpoint(args.baseline_ckpt)
  model_dict = dict(
                baseline=baseline
                )

  Test(model_dict, device=0, save_images=args.save_images).run(args.data_dpath,args.labels_dpath,args.shots_to_test, args.finetune_dict)
