from argparse import ArgumentParser, Namespace
from pathlib import Path
from datetime import datetime, timedelta, timezone

import torch
from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


from datasets import BidDataset
from models import HandsClassifier
from early_stopping import EarlyStoppingWarmup

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/",
    )

    timezone_offset = 8
    tzinfo = timezone(timedelta(hours=timezone_offset))
    current_time = datetime.now(tzinfo).strftime("%Y%m%d_%H%M")
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default=f"./ckpt/hands/{current_time}/",
    )

    parser.add_argument("--rand_seed", type=int, default=1123)

    # data loadser
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)

    # model
    parser.add_argument("--hand_hidden_size", type=int, default=256)
    parser.add_argument("--gru_hidden_size", type=int, default=36)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=.5)
    parser.add_argument("--bidirectional", action="store_true")

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
    )
    parser.add_argument("--num_epoch", type=int, default=100)

    # Fine tuning
    parser.add_argument("--tune", action="store_true")

    # Resume
    parser.add_argument(
        "--resume",
        type=Path,
        help="Model path to resume",
        default=None,
    )

    args = parser.parse_args()
    return args

def train(args: Namespace) -> None:
    train_dataset = BidDataset(args.data_dir / "train.txt")
    valid_dataset = BidDataset(args.data_dir / "valid.txt")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    if args.resume:
        model = HandsClassifier.load_from_checkpoint(args.resume)
        print(f"Load checkpoint from {args.resume}")
    else:
        model = HandsClassifier(
            hand_hidden_size=args.hand_hidden_size,
            gru_hidden_size=args.gru_hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            bidirectional=args.bidirectional,
            lr=args.lr,
            weight_decay=args.weight_decay
        )

    # Train the model
    early_stopping = EarlyStoppingWarmup(
        warmup=10,
        monitor="valid_loss",
        mode="min",
        min_delta=0,
        patience=5,
        check_on_train_epoch_end=False
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="valid_card_acc",
        mode="max",
        dirpath=args.ckpt_dir,
        filename="hands-{epoch:02d}-{valid_card_acc:.2f}-{valid_loss:.2f}",
        save_top_k=1,
        save_on_train_epoch_end=False
    )

    tensorboard_logger = TensorBoardLogger("lightning_logs", name="hands")

    if args.device.type == "cpu":
        trainer = Trainer(
            logger=tensorboard_logger,
            accelerator="cpu",
            deterministic=True,
            max_epochs=args.num_epoch,
            callbacks=[early_stopping, checkpoint_callback],
            # gradient_clip_val=1,
            auto_lr_find=True,
            # profiler="simple"
        )
    else:
        trainer = Trainer(
            logger=tensorboard_logger,
            devices=[args.device.index] if args.device.index else 1,
            accelerator="gpu",
            deterministic=True,
            max_epochs=args.num_epoch,
            callbacks=[early_stopping, checkpoint_callback],
            # gradient_clip_val=1,
            auto_lr_find=True,
            # profiler="simple"
        )

    if args.tune:
        result = trainer.tune(model, train_dataloader, valid_dataloader)
    else:
        trainer.fit(model, train_dataloader, valid_dataloader)
        print(f"Best model saved at {checkpoint_callback.best_model_path}")

if __name__ == '__main__':
    args = parse_args()
    if not args.tune:
        args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.rand_seed)
    train(args)