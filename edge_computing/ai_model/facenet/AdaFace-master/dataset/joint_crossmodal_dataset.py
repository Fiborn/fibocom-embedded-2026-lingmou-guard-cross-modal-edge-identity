import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


DOMAIN_VSDN = 0
DOMAIN_OULU = 1
DOMAIN_HFB = 2

MODALITY_FAKE = 0
MODALITY_REAL = 1

ROLE_IGNORE = -1
ROLE_PROBE = 0
ROLE_GALLERY = 1


def read_annotation_lines(annotation_path):
    with open(annotation_path, "r", encoding="utf-8") as file:
        lines = [line.strip() for line in file if line.strip()]

    if not lines:
        raise ValueError("Annotation file is empty: {}".format(annotation_path))

    return lines


def parse_annotation_line(line):
    parts = line.split(";", 1)

    if len(parts) != 2:
        raise ValueError(
            "Invalid annotation line, expected label;path: {}".format(line)
        )

    return int(parts[0]), parts[1].strip()


def get_domain(path):
    path = path.replace("\\", "/").lower()

    if "/vsdn/" in path:
        return DOMAIN_VSDN

    if "/oulu_256/" in path:
        return DOMAIN_OULU

    if "/hfb_256/" in path:
        return DOMAIN_HFB

    return None


def get_modality(path):
    path = path.lower()

    if path.endswith("_fake.bmp"):
        return MODALITY_FAKE

    if path.endswith("_real.bmp"):
        return MODALITY_REAL

    return None


def load_bgr_tensor(image_path, image_size=112, random_flip=False):
    """
    AdaFace 输入规范：
    - 112 x 112
    - BGR 通道
    - 归一化到 [-1, 1]
    """
    image = Image.open(image_path).convert("RGB")

    if random_flip and random.random() < 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)

    image = image.resize((image_size, image_size), Image.BILINEAR)

    image = np.asarray(image, dtype=np.float32)

    # PIL 读取的是 RGB；AdaFace 预训练模型需要 BGR。
    image = image[:, :, ::-1].copy()

    image = (image / 255.0 - 0.5) / 0.5
    image = np.transpose(image, (2, 0, 1))

    return torch.from_numpy(image).float()


class JointCrossModalTripletDataset(Dataset):
    """
    联合跨模态训练数据集。

    VSDN：
      - 50%：X8_fake(id) -> X1_real(id) -> X1_real(other)
      - 50%：X1-X8 任意 fake(id) -> 任意 real(id) -> real(other)

    Oulu_256 / HFB_256：
      - fake(id) -> real(id) -> real(other)
      - real(id) -> fake(id) -> fake(other)

    返回单个三元组：
      images      : [3, 3, 112, 112]
      labels      : [id, id, other_id]
      modalities  : [fake/real/fake-or-real]
      domain_ids  : [domain, domain, domain]
    """

    def __init__(
        self,
        annotation_lines,
        image_size=112,
        label_map=None,
        vsdn_target_ratio=0.5,
        random_flip=True,
    ):
        self.annotation_lines = annotation_lines
        self.image_size = image_size
        self.label_map = label_map
        self.vsdn_target_ratio = vsdn_target_ratio
        self.random_flip = random_flip

        self.data = {
            DOMAIN_VSDN: {
                "fake": {},
                "real": {},
            },
            DOMAIN_OULU: {
                "fake": {},
                "real": {},
            },
            DOMAIN_HFB: {
                "fake": {},
                "real": {},
            },
        }

        # VSDN 最终目标专用索引
        self.vsdn_target = {
            "fake": {},  # X8 fake
            "real": {},  # X1 real
        }

        self._build_index()

        self.valid_ids = {}

        for domain_id in self.data:
            fake_ids = set(self.data[domain_id]["fake"].keys())
            real_ids = set(self.data[domain_id]["real"].keys())
            valid_ids = sorted(fake_ids & real_ids)

            if len(valid_ids) >= 2:
                self.valid_ids[domain_id] = valid_ids

        if not self.valid_ids:
            raise ValueError("No valid fake-real identities were found.")

        self.domain_ids = sorted(self.valid_ids.keys())

        self._print_summary()

    def _build_index(self):
        for line in self.annotation_lines:
            raw_label, image_path = parse_annotation_line(line)

            domain_id = get_domain(image_path)
            modality = get_modality(image_path)

            if domain_id is None or modality is None:
                continue

            key = "fake" if modality == MODALITY_FAKE else "real"

            self.data[domain_id][key].setdefault(
                raw_label, []
            ).append(image_path)

            normalized_path = image_path.replace("\\", "/").lower()

            if (
                domain_id == DOMAIN_VSDN
                and modality == MODALITY_FAKE
                and "/vsdn/x8/" in normalized_path
            ):
                self.vsdn_target["fake"].setdefault(
                    raw_label, []
                ).append(image_path)

            if (
                domain_id == DOMAIN_VSDN
                and modality == MODALITY_REAL
                and "/vsdn/x1/" in normalized_path
            ):
                self.vsdn_target["real"].setdefault(
                    raw_label, []
                ).append(image_path)

    def _print_summary(self):
        domain_name = {
            DOMAIN_VSDN: "VSDN",
            DOMAIN_OULU: "Oulu_256",
            DOMAIN_HFB: "HFB_256",
        }

        print("\nJoint cross-modal triplet dataset:")

        for domain_id in self.domain_ids:
            valid_ids = self.valid_ids[domain_id]

            fake_count = sum(
                len(self.data[domain_id]["fake"][identity])
                for identity in valid_ids
            )

            real_count = sum(
                len(self.data[domain_id]["real"][identity])
                for identity in valid_ids
            )

            print(
                "  {:<10} identities: {:<3} fake: {:<5} real: {:<5}".format(
                    domain_name[domain_id],
                    len(valid_ids),
                    fake_count,
                    real_count,
                )
            )

    def __len__(self):
        # 每个 epoch 保持与原 FaceNet 设置近似的总图像数量。
        # 一个 Dataset 项目对应一个三元组、三张图。
        return len(self.annotation_lines) // 3

    def _output_label(self, raw_label):
        if self.label_map is None:
            return raw_label

        if raw_label not in self.label_map:
            raise KeyError(
                "Raw label {} is absent from label_map.".format(raw_label)
            )

        return self.label_map[raw_label]

    def _sample_negative_identity(self, domain_id, positive_id):
        candidates = [
            identity
            for identity in self.valid_ids[domain_id]
            if identity != positive_id
        ]

        return random.choice(candidates)

    def _can_use_vsdn_target(self, positive_id, negative_id):
        return (
            positive_id in self.vsdn_target["fake"]
            and positive_id in self.vsdn_target["real"]
            and negative_id in self.vsdn_target["fake"]
            and negative_id in self.vsdn_target["real"]
        )

    def __getitem__(self, index):
        # 三个数据集均衡采样。
        domain_id = self.domain_ids[index % len(self.domain_ids)]

        positive_id = random.choice(self.valid_ids[domain_id])
        negative_id = self._sample_negative_identity(
            domain_id,
            positive_id,
        )

        use_vsdn_target = (
            domain_id == DOMAIN_VSDN
            and self._can_use_vsdn_target(positive_id, negative_id)
            and random.random() < self.vsdn_target_ratio
        )

        if use_vsdn_target:
            fake_path = random.choice(
                self.vsdn_target["fake"][positive_id]
            )
            real_path = random.choice(
                self.vsdn_target["real"][positive_id]
            )
            fake_negative_path = random.choice(
                self.vsdn_target["fake"][negative_id]
            )
            real_negative_path = random.choice(
                self.vsdn_target["real"][negative_id]
            )
        else:
            # VSDN 使用 X1-X8 的所有 fake / real 图像；
            # Oulu、HFB 使用各自所有 fake / real 图像。
            fake_path = random.choice(
                self.data[domain_id]["fake"][positive_id]
            )
            real_path = random.choice(
                self.data[domain_id]["real"][positive_id]
            )
            fake_negative_path = random.choice(
                self.data[domain_id]["fake"][negative_id]
            )
            real_negative_path = random.choice(
                self.data[domain_id]["real"][negative_id]
            )

        # A / B 两个方向交替。
        if index % 2 == 0:
            anchor_path = fake_path
            positive_path = real_path
            negative_path = real_negative_path

            modalities = torch.tensor(
                [MODALITY_FAKE, MODALITY_REAL, MODALITY_REAL],
                dtype=torch.long,
            )
        else:
            anchor_path = real_path
            positive_path = fake_path
            negative_path = fake_negative_path

            modalities = torch.tensor(
                [MODALITY_REAL, MODALITY_FAKE, MODALITY_FAKE],
                dtype=torch.long,
            )

        anchor = load_bgr_tensor(
            anchor_path,
            self.image_size,
            self.random_flip,
        )
        positive = load_bgr_tensor(
            positive_path,
            self.image_size,
            self.random_flip,
        )
        negative = load_bgr_tensor(
            negative_path,
            self.image_size,
            self.random_flip,
        )

        images = torch.stack([anchor, positive, negative], dim=0)

        positive_label = self._output_label(positive_id)
        negative_label = self._output_label(negative_id)

        labels = torch.tensor(
            [positive_label, positive_label, negative_label],
            dtype=torch.long,
        )

        domain_ids = torch.tensor(
            [domain_id, domain_id, domain_id],
            dtype=torch.long,
        )

        return images, labels, modalities, domain_ids


def triplet_collate(batch):
    """
    输出顺序：

    A1, A2, ... , P1, P2, ... , N1, N2, ...

    使每个 anchor 都能在 batch 内找到同身份、异模态 positive。
    """
    anchors = torch.stack([item[0][0] for item in batch], dim=0)
    positives = torch.stack([item[0][1] for item in batch], dim=0)
    negatives = torch.stack([item[0][2] for item in batch], dim=0)

    anchor_labels = torch.stack([item[1][0] for item in batch], dim=0)
    positive_labels = torch.stack([item[1][1] for item in batch], dim=0)
    negative_labels = torch.stack([item[1][2] for item in batch], dim=0)

    anchor_modalities = torch.stack([item[2][0] for item in batch], dim=0)
    positive_modalities = torch.stack(
        [item[2][1] for item in batch],
        dim=0,
    )
    negative_modalities = torch.stack(
        [item[2][2] for item in batch],
        dim=0,
    )

    anchor_domains = torch.stack([item[3][0] for item in batch], dim=0)
    positive_domains = torch.stack([item[3][1] for item in batch], dim=0)
    negative_domains = torch.stack([item[3][2] for item in batch], dim=0)

    return (
        torch.cat([anchors, positives, negatives], dim=0),
        torch.cat(
            [anchor_labels, positive_labels, negative_labels],
            dim=0,
        ),
        torch.cat(
            [anchor_modalities, positive_modalities, negative_modalities],
            dim=0,
        ),
        torch.cat(
            [anchor_domains, positive_domains, negative_domains],
            dim=0,
        ),
    )


class JointRetrievalDataset(Dataset):
    """
    验证/测试检索数据集。

    VSDN：
      Probe   = X8 fake
      Gallery = X1 real

    Oulu/HFB：
      Probe   = fake
      Gallery = real
    """

    def __init__(self, annotation_lines, image_size=112):
        self.image_size = image_size
        self.samples = []

        for line in annotation_lines:
            raw_label, image_path = parse_annotation_line(line)

            domain_id = get_domain(image_path)
            modality = get_modality(image_path)

            if domain_id is None or modality is None:
                continue

            normalized_path = image_path.replace("\\", "/").lower()
            role = ROLE_IGNORE

            if domain_id == DOMAIN_VSDN:
                if (
                    modality == MODALITY_FAKE
                    and "/vsdn/x8/" in normalized_path
                ):
                    role = ROLE_PROBE

                elif (
                    modality == MODALITY_REAL
                    and "/vsdn/x1/" in normalized_path
                ):
                    role = ROLE_GALLERY
            else:
                if modality == MODALITY_FAKE:
                    role = ROLE_PROBE
                elif modality == MODALITY_REAL:
                    role = ROLE_GALLERY

            if role != ROLE_IGNORE:
                self.samples.append(
                    (
                        image_path,
                        raw_label,
                        domain_id,
                        role,
                    )
                )

        if not self.samples:
            raise ValueError("No valid retrieval samples found.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label, domain_id, role = self.samples[index]

        image = load_bgr_tensor(
            image_path,
            self.image_size,
            random_flip=False,
        )

        return (
            image,
            torch.tensor(label, dtype=torch.long),
            torch.tensor(domain_id, dtype=torch.long),
            torch.tensor(role, dtype=torch.long),
        )