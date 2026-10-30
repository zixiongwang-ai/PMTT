"""NYUv2 millimeter depth and ADE20K label conversion."""
import numpy as np
from PIL import Image
from pmtt.settings import data_root

def image_label(row, task):
    with Image.open(data_root(task) / row['image']) as image:
        rgb = image.convert('RGB')
    with Image.open(data_root(task) / row['label']) as image:
        label = np.array(image)
    if label.shape != rgb.size[::-1]:
        raise ValueError('RGB/label size mismatch')
    if task == 'nyu':
        label = label.astype(np.float64) / 1000.
    else:
        label = label.astype(np.int64) - 1
        if not np.all((label >= -1) & (label < 150)):
            raise ValueError('Unknown ADE20K label ID')
    return rgb, label
