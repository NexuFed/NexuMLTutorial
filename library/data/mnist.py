"""MNIST dataset source for in-memory image experiments and benchmarks."""

import numpy as np
import pandas as pd
import torchvision
from nexuml.core.discovery import data_source
from nexuml.data.dataset import NexuDataset
from tensordict import TensorDict


@data_source("MNISTDataset")
class MNISTDataset(NexuDataset):
    LABEL_NAMES = ["digit"]
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
            meta (pd.DataFrame | None, optional): Holds information about every sample in the dataset like path to the file and all labels. Defaults to None.
            label_names (list[str] | None, optional): Names of the labels in the metadata needed for the training process. Defaults to None.
            split_ratio (list[float] | None, optional): Ratios for splitting the dataset. Defaults to None.
            do_split (bool, optional): Whether to split the dataset. Defaults to False.
            modality (str, optional): Modality of the data to load with NVIDIA DALI (instead of using ). Supported are: image, audio. Defaults to "image".
            data (Dataset | Sequence | torch.Tensor | np.ndarray | None, optional): The actual data for the dataset. Defaults to None.
        """

        meta = pd.DataFrame(
            {
                "idx": list(range(60000)),
                "class_labels": np.random.randint(0, 10, size=(60000,)),
            }
        )
        data = torchvision.datasets.MNIST(
            root=root,
            train=train,
            download=download,
            transform=torchvision.transforms.ToTensor(),
        )

        super().__init__(
            data=data,
            meta=meta,
            label_names=self.LABEL_NAMES,
            modality=self.MODALITY,
            **kwargs,
        )

        pass

    def load_item(self, idx: int, row: pd.Series) -> TensorDict:
        return super().load_item(idx, row)

    def load_labels(self, idx: int, row: pd.Series) -> TensorDict | None:
        return super().load_labels(idx, row)
