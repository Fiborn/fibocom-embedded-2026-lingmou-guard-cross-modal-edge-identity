import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from pytorch_lightning import LightningModule
from torch.nn import CrossEntropyLoss

import head
import net
from dataset.joint_crossmodal_dataset import (
    DOMAIN_HFB,
    DOMAIN_OULU,
    DOMAIN_VSDN,
    ROLE_GALLERY,
    ROLE_PROBE,
)


def cross_modal_batch_hard_loss(
    embeddings,
    labels,
    modalities,
    domain_ids,
    margin=0.2,
):
    """
    正样本：
      同身份 + fake/real 异模态 + 同一数据集。

    负样本：
      异身份 + 同一数据集。

    因此 VSDN、Oulu、HFB 不会相互充当负样本。
    """
    embeddings = F.normalize(embeddings.float(), p=2, dim=1)

    distances = torch.cdist(embeddings, embeddings, p=2)

    same_identity = labels[:, None].eq(labels[None, :])
    different_identity = ~same_identity

    same_domain = domain_ids[:, None].eq(domain_ids[None, :])
    different_modality = modalities[:, None].ne(modalities[None, :])

    positive_mask = (
        same_identity
        & same_domain
        & different_modality
    )

    negative_mask = (
        different_identity
        & same_domain
    )

    hardest_positive = distances.masked_fill(
        ~positive_mask,
        float("-inf"),
    ).max(dim=1).values

    hardest_negative = distances.masked_fill(
        ~negative_mask,
        float("inf"),
    ).min(dim=1).values

    valid = (
        positive_mask.any(dim=1)
        & negative_mask.any(dim=1)
    )

    if not valid.any():
        return embeddings.sum() * 0.0

    return F.relu(
        hardest_positive[valid]
        - hardest_negative[valid]
        + margin
    ).mean()


class JointAdaFaceTrainer(LightningModule):
    def __init__(
        self,
        arch="ir_18",
        num_classes=120,
        start_from_model_statedict="",
        lr=1e-3,
        momentum=0.9,
        weight_decay=5e-4,
        lr_milestones=(40, 70, 90),
        lr_gamma=0.1,
        lambda_triplet=1.0,
        lambda_ce=0.25,
        triplet_margin=0.2,
        adaface_m=0.4,
        adaface_h=0.333,
        adaface_s=64.0,
        adaface_t_alpha=0.01,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.model = net.build_model(model_name=arch)

        self.head = head.build_head(
            head_type="adaface",
            embedding_size=512,
            class_num=num_classes,
            m=adaface_m,
            h=adaface_h,
            s=adaface_s,
            t_alpha=adaface_t_alpha,
        )

        self.cross_entropy_loss = CrossEntropyLoss()
        self._validation_outputs = []
        self._test_outputs = []
        if start_from_model_statedict:
            self._load_backbone(start_from_model_statedict)

    def _load_backbone(self, checkpoint_path):
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
        )

        state_dict = checkpoint.get("state_dict", checkpoint)

        backbone_state_dict = {
            key.replace("model.", "", 1): value
            for key, value in state_dict.items()
            if key.startswith("model.")
        }

        if not backbone_state_dict:
            raise ValueError(
                "No backbone weights prefixed by 'model.' were found in {}."
                .format(checkpoint_path)
            )

        missing_keys, unexpected_keys = self.model.load_state_dict(
            backbone_state_dict,
            strict=False,
        )

        print("\nLoaded AdaFace backbone from:", checkpoint_path)
        print("Missing backbone keys:", len(missing_keys))
        print("Unexpected backbone keys:", len(unexpected_keys))
        print("AdaFace classification head is initialized for this task.")

    def forward(self, images):
        return self.model(images)

    def training_step(self, batch, batch_idx):
        images, labels, modalities, domain_ids = batch

        embeddings, norms = self.model(images)

        logits = self.head(
            embeddings,
            norms,
            labels.clone(),
        )

        if isinstance(logits, tuple):
            logits, bad_grad = logits
            labels = labels.clone()
            labels[bad_grad.squeeze(-1)] = -100

        ce_loss = self.cross_entropy_loss(logits, labels)

        triplet_loss = cross_modal_batch_hard_loss(
            embeddings=embeddings,
            labels=labels,
            modalities=modalities,
            domain_ids=domain_ids,
            margin=self.hparams.triplet_margin,
        )

        total_loss = (
            self.hparams.lambda_triplet * triplet_loss
            + self.hparams.lambda_ce * ce_loss
        )

        self.log(
            "train_loss",
            total_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=images.size(0),
        )

        self.log(
            "train_triplet_loss",
            triplet_loss,
            on_step=True,
            on_epoch=True,
            batch_size=images.size(0),
        )

        self.log(
            "train_ce_loss",
            ce_loss,
            on_step=True,
            on_epoch=True,
            batch_size=images.size(0),
        )

        return total_loss


    def on_validation_epoch_start(self):
        self._validation_outputs = []

    def validation_step(self, batch, batch_idx):
        images, labels, domain_ids, roles = batch

        embeddings, _ = self.model(images)
        embeddings = F.normalize(embeddings, p=2, dim=1)

        self._validation_outputs.append(
            {
                "embeddings": embeddings.detach().cpu(),
                "labels": labels.detach().cpu(),
                "domains": domain_ids.detach().cpu(),
                "roles": roles.detach().cpu(),
            }
        )

    def on_validation_epoch_end(self):
        self._evaluate_retrieval(
            self._validation_outputs,
            prefix="val",
        )
        self._validation_outputs = []

    def on_test_epoch_start(self):
        self._test_outputs = []

    def test_step(self, batch, batch_idx):
        images, labels, domain_ids, roles = batch

        embeddings, _ = self.model(images)
        embeddings = F.normalize(embeddings, p=2, dim=1)

        self._test_outputs.append(
            {
                "embeddings": embeddings.detach().cpu(),
                "labels": labels.detach().cpu(),
                "domains": domain_ids.detach().cpu(),
                "roles": roles.detach().cpu(),
            }
        )

    def on_test_epoch_end(self):
        self._evaluate_retrieval(
            self._test_outputs,
            prefix="test",
        )
        self._test_outputs = []
    def _evaluate_retrieval(self, outputs, prefix):
        embeddings = torch.cat(
            [output["embeddings"] for output in outputs],
            dim=0,
        )

        labels = torch.cat(
            [output["labels"] for output in outputs],
            dim=0,
        )

        domains = torch.cat(
            [output["domains"] for output in outputs],
            dim=0,
        )

        roles = torch.cat(
            [output["roles"] for output in outputs],
            dim=0,
        )

        domain_specs = [
            (
                DOMAIN_VSDN,
                "vsdn_x8_x1_rank1",
            ),
            (
                DOMAIN_OULU,
                "oulu_rank1",
            ),
            (
                DOMAIN_HFB,
                "hfb_rank1",
            ),
        ]

        rank1_values = []

        for domain_id, metric_name in domain_specs:
            probe_mask = (
                domains.eq(domain_id)
                & roles.eq(ROLE_PROBE)
            )

            gallery_mask = (
                domains.eq(domain_id)
                & roles.eq(ROLE_GALLERY)
            )

            probe_embeddings = embeddings[probe_mask]
            probe_labels = labels[probe_mask]

            gallery_embeddings = embeddings[gallery_mask]
            gallery_labels = labels[gallery_mask]

            if len(probe_embeddings) == 0 or len(gallery_embeddings) == 0:
                print(
                    "{}: skipped {} because probe/gallery is empty."
                    .format(prefix, metric_name)
                )
                continue

            similarities = torch.mm(
                probe_embeddings,
                gallery_embeddings.t(),
            )

            best_gallery_index = torch.argmax(
                similarities,
                dim=1,
            )

            predicted_labels = gallery_labels[best_gallery_index]

            rank1 = (
                predicted_labels.eq(probe_labels)
                .float()
                .mean()
            )

            self.log(
                "{}_{}".format(prefix, metric_name),
                rank1,
                prog_bar=True,
                logger=True,
            )

            rank1_values.append(rank1)

            print(
                "{} {}: {:.4f} "
                "(probes={}, gallery={})".format(
                    prefix,
                    metric_name,
                    rank1.item(),
                    len(probe_embeddings),
                    len(gallery_embeddings),
                )
            )

        if rank1_values:
            mean_rank1 = torch.stack(rank1_values).mean()

            self.log(
                "{}_mean_rank1".format(prefix),
                mean_rank1,
                prog_bar=True,
                logger=True,
            )

    def configure_optimizers(self):
        decay_parameters = []
        no_decay_parameters = []

        for module in self.model.modules():
            if isinstance(
                module,
                torch.nn.modules.batchnorm._BatchNorm,
            ):
                no_decay_parameters.extend(
                    list(module.parameters())
                )
            elif len(list(module.children())) == 0:
                decay_parameters.extend(
                    list(module.parameters())
                )

        optimizer = optim.SGD(
            [
                {
                    "params": decay_parameters
                    + list(self.head.parameters()),
                    "weight_decay": self.hparams.weight_decay,
                },
                {
                    "params": no_decay_parameters,
                    "weight_decay": 0.0,
                },
            ],
            lr=self.hparams.lr,
            momentum=self.hparams.momentum,
        )

        scheduler = lr_scheduler.MultiStepLR(
            optimizer,
            milestones=list(self.hparams.lr_milestones),
            gamma=self.hparams.lr_gamma,
        )

        return [optimizer], [scheduler]