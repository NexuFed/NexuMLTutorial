"""MNIST dataset source for in-memory image experiments and benchmarks."""

import pandas as pd
import torchvision
from nexuml.core.components import DataSourceDefinition
from nexuml.core.discovery import data_source
from nexuml.data.dataset import NexuDataset


@data_source("MNISTDataset")
class MNISTDataset(DataSourceDefinition):
    root: str = "data/mnist"
    train: bool = True
    download: bool = True

    def build(self) -> NexuDataset:
        return _MNISTDatasetRuntime(**self.model_dump())


class _MNISTDatasetRuntime(NexuDataset):
    LABEL_NAMES = ["class"]
    MODALITY = "image"

    def __init__(
        self,
        root: str = "data/mnist",
        train: bool = True,
        download: bool = True,
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
                self.LABEL_NAMES[0]: data.targets.tolist(),
            }
        )

        super().__init__(
            data=data,
            meta=meta,
            label_names=self.LABEL_NAMES,
            modality=self.MODALITY,
        )
