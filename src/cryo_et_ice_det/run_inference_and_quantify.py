import os
import json
import time
import re
import copy
import argparse
import ast
from datetime import date

import dotenv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

import torch
import torch.nn.functional as F
import torchmetrics
from torch.utils.data import Dataset
from torchvision import transforms as TF
import lightning.pytorch as pl
import segmentation_models_pytorch as smp

from tqdm import tqdm
from IPython import embed

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig

from nn_core.common import PROJECT_ROOT
from pl_modules.baseline import SegModel
from data.dataset_inference import DatasetInference


pl.seed_everything(10)
torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False

dotenv.load_dotenv()
DATA_ROOT = os.environ.get('FULL_DATA_DPATH')

class Test:
    def __init__(self, models, device, save_images, export_annotations):
        self.models = models       
        self.device = f'cuda:{device}' if device is not None else 'cpu'
        self.save_images = save_images
        self.export_annotations = export_annotations

        self.result_dpath = os.path.join(PROJECT_ROOT, 'inference-results', date.today().strftime("%d-%m-%Y"))
        os.makedirs(self.result_dpath, exist_ok=True)
        self.quantification_results = []
        self.results = []
        self.inference_time = 0.0 


    def extract_contours(self,binary_mask):
        contours, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        segmentation = []
        for contour in contours:
            flat_contour = contour.flatten().tolist()
            if len(flat_contour) > 4:  
                segmentation.append(flat_contour)
        return segmentation    

    def save_visualized_images(self, img_tensor, pred_mask_tensor, img_id, folder_path, alpha=0.3):
        contours = None
        if self.save_images or self.export_annotations:
            pred_mask = np.array(pred_mask_tensor.squeeze().cpu() > 0.5)  # Thresholding prediction
            contours = self.extract_contours(pred_mask)
            if self.save_images:
                os.makedirs(folder_path, exist_ok=True)
                img = TF.to_pil_image(1 - img_tensor.squeeze().cpu())
                pred_mask = np.array(pred_mask_tensor.squeeze().cpu() > 0.5) # Thresholding prediction
                contours = self.extract_contours(pred_mask)
                img_np = np.array(img.convert('RGB'))
                pred_mask_np = np.array(pred_mask, dtype=np.uint8) * 255
                pred_colored_mask = np.zeros_like(img_np)
                pred_colored_mask[..., 0] = pred_mask_np  # Red channel for pred mask
                
                combined_img = img_np.copy()
                combined_img[pred_mask_np > 0] = (1-alpha)*combined_img[pred_mask_np > 0] + alpha*pred_colored_mask[pred_mask_np > 0]
                
                plt.imsave(os.path.join(folder_path, f'image_with_mask_{img_id}.png'), combined_img)
        return contours

    def _eval(self, model, X, img_path, img_id, im_dir, total_area):
        X = X.to(self.device)
        start_time = time.time()
        with torch.no_grad():
            out = model(X).squeeze(1)
        end_time = time.time()
        self.inference_time += end_time - start_time

        pix_pred = (out > 0.5).sum().item()
        perc_area = (pix_pred / total_area) * 100

        contours = self.save_visualized_images(X, out, img_id, im_dir)
        ts_id, frame_id = self.parse_image_path(img_path)
        
        self.record_quantification(img_id, img_path, perc_area, ts_id, frame_id)
        
        if self.export_annotations:
            self.record_annotations(img_id, img_path, contours)

    def parse_image_path(self, img_path):
        parts = img_path.split('_')
        ts_id = parts[2]
        frame_id = parts[3].split('.')[0]
        return ts_id, frame_id

    def record_quantification(self, img_id, img_path, perc_area, ts_id, frame_id):
        self.quantification_results.append({
            'id': img_id,
            'file_name': img_path,
            'perc_area': perc_area,
            'ts_id': ts_id,
            'frame_id': frame_id
        })

    def record_annotations(self, img_id, img_path, contours):
        self.results.append({
            'id': img_id,
            'file_name': img_path,
            'num_ann': len(contours) if contours is not None else 0,
            'segmentation': contours if contours is not None else [],
        })

    def run(self):
        dataset = DatasetInference(root_dir=DATA_ROOT)
        total_area = dataset[0][0].numel()
        for model_name, _model in self.models.items():
            _model = _model.to(self.device)
            _model.eval()
            im_dir = os.path.join(self.result_dpath, model_name, 'eval')
            os.makedirs(im_dir, exist_ok=True)
            progress_bar = tqdm(enumerate(dataset), total=len(dataset), desc="Processing images")
            for i, (img) in progress_bar:
                img_path = dataset.file_paths[i]
                self._eval(_model, img, img_path, i, im_dir, total_area)

        print(f"Total inference time: {self.inference_time:.4f} s")
        if len(dataset) > 0:
            avg_time_per_image = self.inference_time / len(dataset)
            print(f"Average inference time per image: {avg_time_per_image * 1000:.1f} ms")  

        if self.export_annotations:
            df = pd.DataFrame(self.results)
            df.to_csv(os.path.join(self.result_dpath, 'predictions.csv'), index=False)
            print(f"Predictions saved to: {os.path.join(self.result_dpath, 'predictions.csv')}")

        # Always save quantifications to a separate CSV
        quantification_df = pd.DataFrame(self.quantification_results)
        quantification_df.to_csv(os.path.join(self.result_dpath, 'quantification.csv'), index=False)
        print(f"Quantification results saved to: {os.path.join(self.result_dpath, 'quantification.csv')}")


def main():
    parser = argparse.ArgumentParser(description="Run inference on selected model.")
    parser.add_argument('--date', type=str, default=date.today().strftime("%d-%m-%Y"), help="Date for the experiment results (format: dd-mm-yyyy). Default is today's date.")
    parser.add_argument('--save_images', action='store_true', help="Flag to save output images with predictions.")
    parser.add_argument('--export_annotations', action='store_true', help="Flag to export annotations in COCO format.")
    parser.add_argument('--device', type=str, default=None, help="Specify CUDA device (e.g., 0, 1,...) or None for CPU.")
    parser.add_argument('--shot', type=str, default=None, help="Specify the shot number directly (e.g., 4 for K4). Overrides the best performance lookup.")
    args = parser.parse_args()

    # Determine the shot number either from the file or command line
    if args.shot is None:
        performance_path = f'k-shot-results/{args.date}/best_f1_performance.txt'
        with open(performance_path, 'r') as f:
            content = f.read()
            match = re.search(r"from baseline at K(\d+)", content)
            if match:
                args.shot = match.group(1)
            else:
                raise ValueError("Could not extract the shot number from the performance file.")
    shot_number = args.shot

    # Compute path to the best performing model checkpoint
    baseline_ckpt = f'k-shot-results/{args.date}/baseline/finetune/model_weights/model_weights_iter_{shot_number}.ckpt'

    # Load the model
    baseline = SegModel.load_from_checkpoint(baseline_ckpt)
    model_dict = {
        'baseline': baseline
    }

    # Set up and run the test
    test = Test(model_dict, device=args.device, save_images=args.save_images, export_annotations=args.export_annotations)
    test.run()

    if args.save_images or args.export_annotations:
        # Saving images or exporting annotations as specified
        print("Images or annotations are being processed as requested.")

    print(f"Inference complete. Results are saved in: {args.date}")

if __name__ == "__main__":
    main()

