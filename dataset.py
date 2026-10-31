from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

class OxfordPetDataset(Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        self.image_dir = root / 'images'
        self.mask_dir = root / 'annotations' / 'trimaps'
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        self.images = list(self.image_dir.glob('*.jpg'))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        mask_path = self.mask_dir / image_path.stem + '.png'

        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)

        return image, mask