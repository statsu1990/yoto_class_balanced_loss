import torch

class ImgDataset(torch.utils.data.Dataset):
    def __init__(self, imgs, labels=None, alb_transform=None):
        self.imgs = imgs
        self.labels = labels
        self.alb_transform = alb_transform

    def __getitem__(self, idx):
        if self.alb_transform is not None:
            img = self.alb_transform(image=self.imgs[idx])['image']

        else:
            img = self.imgs[idx]

        if self.labels is not None:
            return img, self.labels[idx]
        else:
            return img

    def __len__(self):
        return len(self.imgs)

def get_dataloader(dataset, batch_size, shuffle=True):
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
