import torch
from torch.utils.data import Dataset
from PIL import Image

class PoultryDatasetClassification(Dataset):
  def __init__(self, images, labels, transform=None):
    self.images = images
    self.labels = labels
    self.transform = transform

  def __len__(self):
    return len(self.images)

  def __getitem__(self, idx):
    img_name = self.images[idx]
    image = Image.open(img_name).convert('RGB')

    if self.transform:
      image = self.transform(image)

    label = torch.tensor(self.labels[idx], dtype=torch.float32)
    return image, label
