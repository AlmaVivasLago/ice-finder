import ast
import logging
from functools import cached_property, partial
from pathlib import Path
from typing import List, Mapping, Optional

import hydra
import omegaconf

import cv2
import numpy as np
import pandas as pd

import lightning.pytorch as pl
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.v2 as v2


from nn_core.common import PROJECT_ROOT
from nn_core.nn_types import Split
from cryo_et_ice_det.utils.io import read_txt_to_list
from cryo_et_ice_det.data.dataset import ClassIndicesSampler

pylogger = logging.getLogger(__name__)


class MetaData:
    def __init__(self, test: str):
        """The data information the Lightning Module will be provided with.

        This is a "bridge" between the Lightning DataModule and the Lightning Module.
        There is no constraint on the class name nor in the stored information, as long as it exposes the
        `save` and `load` methods.

        The Lightning Module will receive an instance of MetaData when instantiated,
        both in the train loop or when restored from a checkpoint.

        This decoupling allows the architecture to be parametric (e.g. in the number of classes) and
        DataModule/Trainer independent (useful in prediction scenarios).
        MetaData should contain all the information needed at test time, derived from its train dataset.

        Examples are the class names in a classification task or the vocabulary in NLP tasks.
        MetaData exposes `save` and `load`. Those are two user-defined methods that specify
        how to serialize and de-serialize the information contained in its attributes.
        This is needed for the checkpointing restore to work properly.

        Args:
            class_vocab: association between class names and their indices
        """
        # example
        self.test: str = test

    def save(self, dst_path: Path) -> None:
        """Serialize the MetaData attributes into the zipped checkpoint in dst_path.

        Args:
            dst_path: the root folder of the metadata inside the zipped checkpoint
        """
        pylogger.debug(f"Saving MetaData to '{dst_path}'")

        # example
        (dst_path / "test.txt").write_text(self.test)

    @staticmethod
    def load(src_path: Path) -> "MetaData":
        """Deserialize the MetaData from the information contained inside the zipped checkpoint in src_path.

        Args:
            src_path: the root folder of the metadata inside the zipped checkpoint

        Returns:
            an instance of MetaData containing the information in the checkpoint
        """
        pylogger.debug(f"Loading MetaData from '{src_path}'")

        # example
        lines = (src_path / "test.txt").read_text(encoding="utf-8")

        return MetaData(
            test=lines,
        )

    def __repr__(self) -> str:
        attributes = ",\n    ".join([f"{key}={value}" for key, value in self.__dict__.items()])
        return f"{self.__class__.__name__}(\n    {attributes}\n)"


def collate_fn(samples: List, split: Split, metadata: MetaData):
    """Custom collate function for dataloaders with access to split and metadata.

    Args:
        samples: A list of samples coming from the Dataset to be merged into a batch
        split: The data split (e.g. train/val/test)
        metadata: The MetaData instance coming from the DataModule or the restored checkpoint

    Returns:
        A batch generated from the given samples
    """
    return samples


class MyDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset: DictConfig,
        num_workers: DictConfig,
        batch_size: DictConfig,
        accelerator: str,
        data_csv_fpath: str,
        classes_csv_fpath: str,
        train_split_fpath: str,
        val_split_fpath: str,
        test_split_fpath: str,
        num_way: int,
        num_support: int,
        num_query: int,
        num_test_tasks: int,
        train_max_steps: int
    ):
        super().__init__()
        self.dataset = dataset
        self.num_workers = num_workers
        self.batch_size = batch_size
        # https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#gpus
        self.pin_memory: bool = accelerator is not None and str(accelerator) == "gpu"

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

        self.data_csv_fpath = data_csv_fpath
        self.classes_csv_fpath = classes_csv_fpath
        self.train_split_fpath = train_split_fpath
        self.val_split_fpath = val_split_fpath
        self.test_split_fpath = test_split_fpath
        self.num_way = num_way
        self.num_support = num_support
        self.num_query = num_query
        self.num_test_tasks = num_test_tasks
        self.train_max_steps = train_max_steps

        self.max_samples_per_class = None

    def _convert_str_to_list(self, val):
        if pd.isna(val):
            return val
        try:
            return ast.literal_eval(val)
        except Exception as e:
            raise Exception("Error while converting string to list:" + e)

    @cached_property
    def metadata(self) -> MetaData:
        """Data information to be fed to the Lightning Module as parameter.

        Examples are vocabularies, number of classes...

        Returns:
            metadata: everything the model should know about the data, wrapped in a MetaData object.
        """

        return MetaData(test="What is this?")

    def prepare_data(self) -> None:
        # download only
        pass

    def setup(self, stage: Optional[str] = None):
        data_df = pd.read_csv(self.data_csv_fpath, index_col=0) 
        data_df['segmentation'] = data_df.segmentation.apply(self._convert_str_to_list)
        
        # Merge instances with segmentation so they are in a single row
        classes_df = pd.read_csv(self.classes_csv_fpath, index_col=0) 
        self.train_idxs = read_txt_to_list(self.train_split_fpath, cast=int)
        self.val_idxs = read_txt_to_list(self.val_split_fpath, cast=int)
        self.test_idxs = read_txt_to_list(self.test_split_fpath, cast=int)

        self.max_samples_per_class = data_df.groupby(['group_id']).size().min()
        #assert self.num_support + self.num_query <= self.max_samples_per_class, f"Error: support + query is {self.num_support + self.num_query}, max samples per class is {self.max_samples_per_class}"

        # Here you should instantiate your dataset, you may also split the train into train and validation if needed.
        if (stage is None or stage == "fit") and (self.train_dataset is None and self.val_dataset is None):
            transforms = False
            self.train_dataset = hydra.utils.instantiate(self.dataset, data_df=data_df, classes_df=classes_df, num_support=self.num_support, num_query=self.num_query, transforms=transforms)
            self.val_dataset = hydra.utils.instantiate(self.dataset, data_df=data_df, classes_df=classes_df, num_support=self.num_support, num_query=self.num_query, transforms=transforms)
        if stage is None or stage == "test":
            self.test_dataset = hydra.utils.instantiate(self.dataset, data_df=data_df, classes_df=classes_df, num_support=self.num_support, num_query=self.num_query)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size.train,
            num_workers=self.num_workers.train,
            pin_memory=self.pin_memory,
            collate_fn=partial(collate_fn, split="train", metadata=self.metadata),
            sampler=ClassIndicesSampler(
                split_idxs=self.train_idxs,
                num_way=self.num_way,
                num_tasks=self.batch_size.train * self.train_max_steps # TODO; take into account when training is being resumed
            ),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            batch_size=self.batch_size.val,
            num_workers=self.num_workers.val,
            pin_memory=self.pin_memory,
            collate_fn=partial(collate_fn, split="val", metadata=self.metadata),
            sampler=ClassIndicesSampler(
                split_idxs=self.val_idxs,
                num_way=self.num_way,
                num_tasks=self.batch_size.val * 4
            ),
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            shuffle=False,
            batch_size=self.batch_size.test,
            num_workers=self.num_workers.test,
            pin_memory=self.pin_memory,
            collate_fn=partial(collate_fn, split="test", metadata=self.metadata),
            sampler=ClassIndicesSampler(
                split_idxs=self.test_idxs,
                num_way=self.num_way,
                num_tasks=self.num_test_tasks
            ),
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(" f"{self.dataset=}, " f"{self.num_workers=}, " f"{self.batch_size=})"


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig) -> None:
    """Debug main to quickly develop the DataModule.

    Args:
        cfg: the hydra configuration
    """
    import time
    
    m: pl.LightningDataModule = hydra.utils.instantiate(cfg.nn.data, _recursive_=False)
    m.metadata
    m.num_workers.train = 0
    m.setup()

    start = time.time()
    batch = next(iter(m.train_dataloader()))
    elapsed = time.time() - start
    print(batch)
    print(f'took {elapsed} s')


if __name__ == "__main__":
    main()
