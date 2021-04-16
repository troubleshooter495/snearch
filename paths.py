import random
import os


def img_paths(folder, rand_resize=lambda: random.randint(1, 100) > 5):
    img_paths = []
    for cat in os.listdir(folder):
        folder1 = folder + "/" + cat
        if cat == '.DS_Store':
            continue
        for subcat in os.listdir(folder1):
            folder2 = folder1 + "/" + subcat
            if subcat == '.DS_Store':
                continue
            for brand in os.listdir(folder2):
                folder3 = folder2 + "/" + brand
                if brand == '.DS_Store':
                    continue
                for pic in os.listdir(folder3):
                    if rand_resize():
                        continue
                    img_paths.append(folder3 + "/" + pic)

    return img_paths
