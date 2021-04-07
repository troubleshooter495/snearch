import numpy as np
import pandas as pd
import torch
from skimage import io
import torchvision.transforms as transforms
import hnswlib
import tqdm
import re
import os
import random
from collections import defaultdict


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


def knn_samples(csv_path, img_paths, model):
    """
        Input:
            csv_path (string): Csv file with labels.
            img_paths (list): List of all images' path names.
            model (torch.nn.Module): autoencoder
        Output:
            samples: array of encoded images
            labels: array of labels for samples
    """
    samples = []
    labels = []

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    csv = pd.read_csv(csv_path)

    for i in tqdm.tqdm(img_paths):
        img = io.imread(i)
        if img.shape != (102, 136, 3) or \
                i == "ut-zap50k-images/Boots/Mid-Calf/Primigi Kids/8022042.89.jpg":
            continue

        img = transform(img).reshape((1, 3, 102, 136))
        with torch.no_grad():
            img = model.encode(img)

        samples.append(torch.flatten(img))
        labels.append(get_label(i, csv))

    for i in range(len(samples)):
        samples[i] = samples[i].numpy()

    samples = np.array(samples)

    print('loaded samples')
    return samples, labels, img_paths


def count_distances(samples, n_top):
    num_elements, dim = samples.shape

    data1 = samples[:num_elements // 2]
    data2 = samples[num_elements // 2:]

    p = hnswlib.Index(space='l2', dim=dim)
    p.init_index(max_elements=num_elements // 2, ef_construction=100, M=16)

    p.set_ef(10)

    p.set_num_threads(4)

    print("Adding first batch of %d elements" % (len(data1)))
    p.add_items(data1)

    labels, distances = p.knn_query(data1, k=n_top)
    print("Recall for the first batch:",
          np.mean(labels.reshape(-1) == np.arange(len(data1))), "\n")

    index_path = 'first_half.bin'
    print("Saving index to '%s'" % index_path)
    p.save_index(index_path)
    del p

    p = hnswlib.Index(space='l2', dim=dim)

    print("\nLoading index from 'first_half.bin'\n")

    p.load_index("first_half.bin", max_elements=num_elements)

    print("Adding the second batch of %d elements" % (len(data2)))
    p.add_items(data2)

    labels, distances = p.knn_query(samples, k=n_top)
    print("Recall for two batches:",
          np.mean(labels.reshape(-1) == np.arange(len(samples))), "\n")
    print('counted cnn')
    return labels, distances
