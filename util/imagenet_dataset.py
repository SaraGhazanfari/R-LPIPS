import os
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder


def get_dataset(batch_size, num_workers, data_path, split='val', shuffle=False, drop_last=False, pin_memory=True,
                has_normalize=True):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    with_normalization_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    without_normalization_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )
    if has_normalize:
        dataset = ImageFolder(root=os.path.join(data_path, split), transform=with_normalization_transform)
    else:
        dataset = ImageFolder(root=os.path.join(data_path, split), transform=without_normalization_transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,  # may need to reduce this depending on your GPU
        num_workers=num_workers,  # may need to reduce this depending on your num of CPUs and RAM
        shuffle=shuffle,
        drop_last=drop_last,
        pin_memory=pin_memory
    )

    return dataset, dataloader
