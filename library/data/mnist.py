"""MNIST dataset source for in-memory image experiments and benchmarks."""

import pandas as pd
import torchvision
from nexuml.core.discovery import data_source
from nexuml.data.dataset import NexuDataset


@data_source("MNISTDataset")
class MNISTDataset(NexuDataset):
    LABEL_NAMES = ["class_labels"]
    MODALITY = "image"

    def __init__(
        self,
        root: str = "data/mnist",
        train: bool = True,
        download: bool = True,
        **kwargs,
    ):
        """MNIST Dataset

        Args:
            root (str, optional): Directory holding the raw MNIST files. Defaults to "data/mnist".
            train (bool, optional): Load the 60000-sample train split, or the 10000-sample test split. Defaults to True.
            download (bool, optional): Download MNIST to `root` if not already present. Defaults to True.
        """

        data = torchvision.datasets.MNIST(
            root=root,
            train=train,
            download=download,
            transform=torchvision.transforms.ToTensor(),
        )

        meta = pd.DataFrame(
            {
                "idx": list(range(len(data))),
                "class_labels": data.targets.tolist(),
            }
        )

        super().__init__(
            data=data,
            meta=meta,
            label_names=self.LABEL_NAMES,
            modality=self.MODALITY,
            **kwargs,
        )
