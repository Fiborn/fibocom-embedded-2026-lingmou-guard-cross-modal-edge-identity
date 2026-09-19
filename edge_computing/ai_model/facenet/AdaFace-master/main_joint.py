import argparse
import os
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from data_joint import JointDataModule
from train_val_joint import JointAdaFaceTrainer


def build_parser():
    project_root = Path(__file__).resolve().parent
    annotation_root = project_root.parent / "cls_train_joint"

    parser = argparse.ArgumentParser(
        description="AdaFace joint cross-modal transfer learning"
    )

    parser.add_argument(
        "--train_txt",
        type=str,
        default=str(annotation_root / "train.txt"),
    )

    parser.add_argument(
        "--val_txt",
        type=str,
        default=str(annotation_root / "val.txt"),
    )

    parser.add_argument(
        "--test_txt",
        type=str,
        default=str(annotation_root / "test.txt"),
    )

    parser.add_argument(
        "--start_from_model_statedict",
        type=str,
        default="adaface_ir18_webface4m.ckpt",
    )

    parser.add_argument(
        "--arch",
        type=str,
        default="ir_18",
        choices=["ir_18", "ir_34", "ir_50", "ir_101", "ir_se_50"],
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./experiments/adaface_ir18_webface4m_joint",
    )

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--image_size", type=int, default=112)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_16bit", action="store_true")

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-4)

    parser.add_argument(
        "--lr_milestones",
        type=str,
        default="40,70,90",
    )

    parser.add_argument("--lr_gamma", type=float, default=0.1)

    parser.add_argument(
        "--lambda_triplet",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--lambda_ce",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--triplet_margin",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--vsdn_target_ratio",
        type=float,
        default=0.5,
        help="VSDN 中 X8_fake -> X1_real 专项三元组比例。",
    )

    parser.add_argument("--adaface_m", type=float, default=0.4)
    parser.add_argument("--adaface_h", type=float, default=0.333)
    parser.add_argument("--adaface_s", type=float, default=64.0)
    parser.add_argument("--adaface_t_alpha", type=float, default=0.01)

    return parser


def main():
    args = build_parser().parse_args()

    if args.batch_size % 3 != 0:
        raise ValueError(
            "--batch_size must be divisible by 3."
        )

    if not 0.0 <= args.vsdn_target_ratio <= 1.0:
        raise ValueError(
            "--vsdn_target_ratio must be between 0 and 1."
        )

    if not os.path.isfile(args.start_from_model_statedict):
        raise FileNotFoundError(
            "AdaFace checkpoint does not exist: {}".format(
                args.start_from_model_statedict
            )
        )

    os.makedirs(args.output_dir, exist_ok=True)

    pl.seed_everything(args.seed, workers=True)

    lr_milestones = [
        int(value)
        for value in args.lr_milestones.split(",")
        if value.strip()
    ]

    data_module = JointDataModule(
        train_txt=args.train_txt,
        val_txt=args.val_txt,
        test_txt=args.test_txt,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        vsdn_target_ratio=args.vsdn_target_ratio,
    )

    model = JointAdaFaceTrainer(
        arch=args.arch,
        num_classes=data_module.num_classes,
        start_from_model_statedict=args.start_from_model_statedict,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        lr_milestones=lr_milestones,
        lr_gamma=args.lr_gamma,
        lambda_triplet=args.lambda_triplet,
        lambda_ce=args.lambda_ce,
        triplet_margin=args.triplet_margin,
        adaface_m=args.adaface_m,
        adaface_h=args.adaface_h,
        adaface_s=args.adaface_s,
        adaface_t_alpha=args.adaface_t_alpha,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.output_dir,
        filename=(
            "epoch{epoch:03d}-"
            "vsdn_rank1{val_vsdn_x8_x1_rank1:.4f}"
        ),
        monitor="val_vsdn_x8_x1_rank1",
        mode="max",
        save_top_k=3,
        save_last=True,
    )

    logger = CSVLogger(
        save_dir=args.output_dir,
        name="logs",
    )

    trainer_kwargs = {
        "default_root_dir": args.output_dir,
        "logger": logger,
        "max_epochs": args.epochs,
        "callbacks": [checkpoint_callback],
        "precision": 16 if args.use_16bit else 32,
        "num_sanity_val_steps": 0,
        "log_every_n_steps": 20,
    }

    # 兼容旧版和新版 PyTorch Lightning。
    try:
        trainer = pl.Trainer(
            accelerator="gpu" if args.gpus > 0 else "cpu",
            devices=args.gpus if args.gpus > 0 else 1,
            **trainer_kwargs
        )
    except TypeError:
        trainer = pl.Trainer(
            gpus=args.gpus,
            **trainer_kwargs
        )

    print("\nStart AdaFace joint cross-modal training.")
    trainer.fit(model, datamodule=data_module)

    print("\nBest validation checkpoint:")
    print(checkpoint_callback.best_model_path)

    print("\nEvaluate once on held-out test.txt.")
    trainer.test(
        model=model,
        datamodule=data_module,
        ckpt_path="best",
    )


if __name__ == "__main__":
    main()