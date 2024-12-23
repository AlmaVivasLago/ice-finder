import argparse
import os
import re
import json

import numpy as np
import pandas as pd

def coco_to_csv(coco_fpath, csv_fpath=None):
  lam_id_regex = r'\d+$'
  just_digits_regex = r'^\d+$'
  frame_id_regex = r'(\d+)\.tiff$'

  with open(coco_fpath, 'r') as fid:
    data = json.load(fid)

  img_df = pd.DataFrame(data['images'])
  ann_df = pd.DataFrame(data['annotations'])
  # Remove unused columns
  img_df = img_df.drop(columns=['license', 'flickr_url', 'coco_url', 'date_captured'])
  ann_df = ann_df.drop(columns=['id', 'category_id', 'iscrowd', 'attributes'])

  # Merge multiple segmentation of a same image_id into a single row
  def merge(x):
    if x.name == 'segmentation':
      return x.apply(lambda x: x[0]).tolist()
    elif x.name == 'area' or x.name == 'bbox':
      return x.tolist()
    else:
      raise Exception('Error: {x.name} is not expected.')
  ann_df = ann_df.groupby('image_id').agg(merge).reset_index()

  # Merge annotations with the rest
  data_df = pd.merge(img_df, ann_df, how='left', left_on='id', right_on='image_id')
  # Drop unused columns
  data_df = data_df.drop(columns=['id', 'image_id'])

  # Extract info from file name
  file_name_info = data_df.file_name.str.split('_')
  data_df['lam_id'] = file_name_info.apply(lambda x: int(re.findall(lam_id_regex, x[0])[0]))
  data_df['ts_id'] = file_name_info.apply(lambda x: int(x[2]))
  data_df['frame_id'] = file_name_info.apply(lambda x: int(re.findall(frame_id_regex, x[3])[0]))
  # Add column that counts the number of segmentations
  data_df['num_ann'] = data_df.segmentation.apply(lambda x: 0 if x is np.nan else len(x))
  
  if csv_fpath is None:
    csv_fpath = os.path.join(coco_fpath.replace('instances_default.json', 'data.csv'))
  data_df.to_csv(csv_fpath)
  print(f"Data saved to {csv_fpath}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert COCO JSON to CSV format.')
    parser.add_argument('--coco_fpath', default='./data/selected/annotations/instances_default.json', help='Path to COCO JSON file.')
    args = parser.parse_args()

    coco_to_csv(args.coco_fpath)