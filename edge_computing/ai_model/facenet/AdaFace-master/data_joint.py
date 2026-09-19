import pytorch_lightning as pl
from torch.utils.data import DataLoader

from dataset.joint_crossmodal_dataset import (
    JointCrossModalTripletDataset,
    JointRetrievalDataset,
    read_annotation_lines,
    triplet_collate,
)


class JointDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_txt,
        val_txt,
        test_txt,
        batch_size=24,
        eval_batch_size=128,
        num_workers=4,
        image_size=112,
        vsdn_target_ratio=0.5,
    ):
        super().__init__()

        if batch_size % 3 != 0:
            raise ValueError(
                "batch_size must be divisible by 3 for triplet training."
            )

        self.train_txt = train_txt
        self.val_txt = val_txt
        self.test_txt = test_txt

        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.vsdn_target_ratio = vsdn_target_ratio

        self.train_lines = read_annotation_lines(train_txt)
        self.val_lines = read_annotation_lines(val_txt)
        self.test_lines = read_annotation_lines(test_txt)

        self.train_raw_labels = sorted(
            {
                int(line.split(";", 1)[0])
                for line in self.train_lines
            }
        )

        self.val_raw_labels = {
            int(line.split(";", 1)[0])
            for line in self.val_lines
        }

        self.test_raw_labels = {
            int(line.split(";", 1)[0])
            for line in self.test_lines
        }

        if set(self.train_raw_labels) & self.val_raw_labels:
            raise ValueError("Training and validation identities overlap.")

        if set(self.train_raw_labels) & self.test_raw_labels:
            raise ValueError("Training and test identities overlap.")

        if self.val_raw_labels & self.test_raw_labels:
            raise ValueError("Validation and test identities overlap.")

        self.train_label_map = {
            raw_label: mapped_label
            for mapped_label, raw_label in enumerate(
                self.train_raw_labels
            )
        }

        self.num_classes = len(self.train_label_map)

        print("\nJoint identity split:")
        print(
            "  Train: {} identities, {} images".format(
                len(self.train_raw_labels),
                len(self.train_lines),
            )
        )
        print(
            "  Val:   {} identities, {} images".format(
                len(self.val_raw_labels),
                len(self.val_lines),
            )
        )
        print(
            "  Test:  {} identities, {} images".format(
                len(self.test_raw_labels),
                len(self.test_lines),
            )
        )
        print("  Classifier classes: {}".format(self.num_classes))

    def prepare_data(self):
        # 不使用官方 AgeDB/CFP/LFW 验证集。
        pass

    def setup(self, stage=None):
        if stage in (None, "fit"):
            self.train_dataset = JointCrossModalTripletDataset(
                annotation_lines=self.train_lines,
                image_size=self.image_size,
                label_map=self.train_label_map,
                vsdn_target_ratio=self.vsdn_target_ratio,
                random_flip=True,
            )

            self.val_dataset = JointRetrievalDataset(
                annotation_lines=self.val_lines,
                image_size=self.image_size,
            )

        if stage in (None, "test"):
            self.test_dataset = JointRetrievalDataset(
                annotation_lines=self.test_lines,
                image_size=self.image_size,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size // 3,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=triplet_collate,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )