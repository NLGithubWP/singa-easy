import random
import os
import shutil
import traceback
import zipfile
import tempfile
import numpy as np
import pandas as pd


def getXrayData(dataset_path):
    dataset_zipfile = zipfile.ZipFile(dataset_path, 'r')
    if 'images.csv' in dataset_zipfile.namelist():
        with tempfile.TemporaryDirectory() as d:
            for fileName in dataset_zipfile.namelist():
                if fileName.endswith('.csv'):
                    images_csv_path = dataset_zipfile.extract(fileName, path=d)
                    break
            try:
                csv = pd.read_csv(images_csv_path)
                image_classes = csv[csv.columns[1:]]
                image_paths = csv[csv.columns[0]]
            except:
                traceback.print_stack()
                raise
        num_classes = len(csv[csv.columns[1]].unique())
        num_labeled_samples = len(csv[csv.columns[0]].unique())
        image_classes = tuple(np.array(image_classes).squeeze().tolist())
        image_paths = tuple(image_paths)

    else:
        image_paths = [
            x for x in dataset_zipfile.namelist()
            if x.endswith('/') == False
        ]
        num_labeled_samples = len(image_paths)
        str_labels = [os.path.dirname(x) for x in image_paths]
        str_labels_set = list(set(str_labels))
        num_classes = len(str_labels_set)
        image_classes = [str_labels_set.index(x) for x in str_labels]

    for i in range(len(image_paths)):
        try:
            if image_classes[i] == 0:
                shutil.copy(
                    os.path.join("/Users/nailixing/Downloads/data/val", image_paths[i]),
                    os.path.join("/Users/nailixing/Downloads/data/val_xray", 'healthy'))
            elif image_classes[i] == 1:
                shutil.copy(
                    os.path.join("/Users/nailixing/Downloads/data/val", image_paths[i]),
                    os.path.join("/Users/nailixing/Downloads/data/val_xray", 'unhealthy'))
        except:
            pass

    return image_paths, image_classes, num_labeled_samples, num_classes


if __name__ == "__main__":
    print(getXrayData("/Users/nailixing/Downloads/data/val.zip"))
