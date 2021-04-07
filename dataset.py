import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from skimage import io
import re
import torch
import tqdm
import random
import os
from collections import defaultdict
import torchvision.transforms as transforms


def get_label(img_path, meta):
    crop = re.search(r"/[0-9]+.[0-9]+", img_path).group(0)[1:]
    code = crop.replace('.', '-')
    label = "Other"
    try:
        label = ", ".join(np.array(meta.loc[meta['CID'] == code])[0][1:3])
    except:
        print("Error in label at", img_path, code,
              np.array(meta.loc[meta['CID'] == code]))
    return label


class ShoesDataset(Dataset):
    def __init__(self, csv_path, img_folder, rand_resize=lambda: random.randint(1, 100) > 5):
        """
        Args:
            csv_path (string): Csv file with labels.
            img_paths (list): List of all images' path names.
            transform (callable): Optional transform to be applied
                on a sample.
        """
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        folder = img_folder
        classes = {}
        numofpics = defaultdict(int)
        img_paths = []
        count = 0

        for cat in os.listdir(folder):
            folder1 = folder + "/" + cat
            if cat == '.DS_Store':
                continue
            for subcat in os.listdir(folder1):
                folder2 = folder1 + "/" + subcat
                classes[cat + ", " + subcat] = count
                classes[count] = cat + ", " + subcat
                count += 1
                if subcat == '.DS_Store':
                    continue
                for brand in os.listdir(folder2):
                    folder3 = folder2 + "/" + brand
                    if brand == '.DS_Store':
                        continue
                    for pic in os.listdir(folder3):
                        if rand_resize():
                            continue
                        numofpics[cat + ", " + subcat] += 1
                        img_paths.append(folder3 + "/" + pic)

        temp = numofpics.copy()
        temp["Other"] = 0
        for key, val in numofpics.items():
            if val < 1000:
                temp["Other"] += val
                temp.pop(key)
                classes.pop(key)
                classes[key] = 22

        classes[22] = "Other"
        classes["Other"] = 22
        numofpics = temp.copy()

        id2label = {x: i for i, x in
                    enumerate(classes[x] for x in numofpics.keys())}
        print("img_paths loaded")

        self.csv = pd.read_csv(csv_path)
        self.transform = transform
        tmpi = []
        self.labels = []
        for i in tqdm.tqdm(img_paths):
            img = io.imread(i)
            if img.shape != (102, 136, 3):
                continue
            self.labels.append(id2label[classes[get_label(i, self.csv)]])
            tmpi.append(transform(img))

        self.imgs = torch.stack(tmpi)
        print("dataset loaded")
        self.img_paths = img_paths

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return [self.imgs[idx], self.labels[idx]]
