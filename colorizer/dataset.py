from torch.utils.data import Dataset
from torchvision import io
from torchvision.transforms import v2


class ImageDataset(Dataset):
    """Torch dataset for loading images from their file paths

    Images are loaded in RGB colorspace, resized and normalized
    between 0 and 1

    Parameters
    ----------
    paths : list[str]
        Images file paths

    resize : int
        Images are resized, where h == w
    """

    def __init__(self, paths: list[str], resize: int):
        self.paths = paths

        self.transform = v2.Compose(
            [
                v2.Resize((resize, resize), antialias=True),
                v2.RandomHorizontalFlip(),
                v2.Lambda(lambda im: im / 255),
            ]
        )

    def __len__(self):
        """Total images paths added to the dataset

        Returns
        -------
        Total images available
        """
        return len(self.paths)

    def __getitem__(self, index):
        """gets an image from the dataset.
        Images are grouped in batches for model consumption

        Parameters
        ----------
        index : int
            Index of the file path of an image for the total availble images
            in `self.paths`

        Returns
        -------
        dict pointing to the loaded image
        """
        image = io.read_image(path=self.paths[index], mode=io.image.ImageReadMode.RGB)
        image = self.transform(image)

        return {"image": image}
