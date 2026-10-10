import os, glob, re
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as T
from sklearn.model_selection import train_test_split

def parse_num(val, default=0.0):
    if pd.isnull(val): return default
    cleaned = re.sub(r'[^\d.]', '', str(val))
    try: return float(cleaned) if cleaned else default
    except ValueError: return default

def load_uhcs_splits(data_dir, excel_path, test_size=0.30, val_ratio=0.50, seed=42):
    cropped_dir = os.path.join(data_dir, 'For Training', 'Cropped')
    disk_files = {os.path.basename(f).lower(): f for f in glob.glob(f"{cropped_dir}/*.png")}
    df = pd.read_excel(excel_path).dropna(subset=['anneal_temperature', 'anneal_time']).copy()

    records = []
    for _, row in df.iterrows():
        raw_name = str(row['path']).strip().lower()
        path = disk_files.get(raw_name) or disk_files.get(f"cropped{raw_name}")
        if path:
            records.append({
                'img_path': path,
                'target': parse_num(row['anneal_temperature']),
                'anneal_time': parse_num(row['anneal_time']),
                'magnification': parse_num(row.get('magnification'), default=5000.0),
                'cool_method': str(row.get('cool_method', 'unknown')).strip().lower(),
                'microconstituent': str(row.get('primary_microconstituent', 'unknown')).strip().lower()
            })

    full_df = pd.DataFrame(records)
    train_df, temp_df = train_test_split(full_df, test_size=test_size, random_state=seed)
    val_df, test_df = train_test_split(temp_df, test_size=val_ratio, random_state=seed)
    return train_df, val_df, test_df

class MicrographDataset(Dataset):
    def __init__(self, paths, transform=None):
        self.paths = paths
        self.tf = transform or T.Compose([
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Resize((224, 224), antialias=True),
            T.Normalize(mean=[0.5], std=[0.25])
        ])
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        with Image.open(self.paths[idx]) as im:
            return self.tf(im.convert('L'))
