from torch.utils.data import Dataset
from torchvision import io
from torchvision.transforms import v2


class ImageDataset(Dataset):
    def __init__(self, paths: list[str], resize: int):
        self.paths = paths

        self.transform = v2.Compose(
            [
                v2.Resize((resize, resize), antialias=False),
                v2.RandomHorizontalFlip(),
                v2.Lambda(lambda im: im / 255),
            ]
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image = io.read_image(path=self.paths[index], mode=io.image.ImageReadMode.RGB)
        image = self.transform(image)

        return {"image": image}
