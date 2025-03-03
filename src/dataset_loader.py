import os
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from PIL import Image

class CustomDataset(Dataset):
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith(("jpg", "png"))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image

def get_dataloader(dataset_path, batch_size=16, image_size=64):
    
    transform = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    if any(os.path.isdir(os.path.join(dataset_path, d)) for d in os.listdir(dataset_path)):
        dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
    else:
        dataset = CustomDataset(root_dir=dataset_path, transform=transform)

    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
