import torch
import argparse
import torch.nn as nn
import torch.optim as optim
from argparse import ArgumentParser

from models.MultiClassificationModel import MultiClassificationModel, ResNetPredictor
from utils.experiment import get_loader, save_model, load_model, train_one_epoch, evaluate_one_epoch, \
    initiate_environment
from models.resnet import resnet18

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Meta info
    parser.add_argument("--task_name", type=str, default="baseline2", help="Task name to save.")
    parser.add_argument("--mode", type=str, choices=["train", "test"], default="test", help="Mode to run.")
    parser.add_argument("--device", type=int, default=torch.device('cuda') if torch.cuda.is_available() else "cpu", help="Device number.")
    parser.add_argument("--num_workers", type=int, default=7, help="Spawn how many processes to load data.")
    parser.add_argument("--rng_seed", type=int, default=142857, help='manual seed')

    # Training
    parser.add_argument("--num_epoch", type=int, default=0, help="Current epoch number.")
    parser.add_argument("--max_epoch", type=int, default=10, help="Max epoch number to run.")
    parser.add_argument("--checkpoint_path", type=str, default="./save/baseline2/ckpt_epoch_last.pth", help="Checkpoint path to load.")
    parser.add_argument("--save_path", type=str, default="./save/", help="Checkpoint path to save.")
    parser.add_argument("--save_freq", type=int, default=1, help="Save model every how many epochs.")
    # TODO Start: Define `args.val_freq` and `args.print_freq` here #
    parser.add_argument("--val_freq", type=int, default=1, help="Validate model every how many epochs.")
    parser.add_argument("--print_freq", type=int, default=1, help="Print loss every how many epochs.")
    # TODO End #
    parser.add_argument("--batch_size", type=int, default=8, help="Entry numbers every batch.")

    #Model
    parser.add_argument("--use_model", type=str, default="VGG16", choices=["VGG16", "ResNet"], help="Model to use.")

    # Optimizer
    parser.add_argument("--optimizer", type=str, choices=["SGD", "Adam"], default="Adam", help="Optimizer type.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for SGD optimizer.")
    parser.add_argument("--weight_decay", type=float, default=0.02, help="Weight decay regularization for model.")

    args = parser.parse_args()
    initiate_environment(args)

    # Prepare dataloader
    loader, val_loader = get_loader(args)

    # Load model & optimizer
    if args.use_model == "VGG16":
        model = MultiClassificationModel()
    elif args.use_model == "ResNet":
        model = ResNetPredictor()
    if args.optimizer == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "Adam":
        # TODO Start: define Adam optimizer here #
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        # TODO End #
    else:
        raise NotImplementedError("You must specify a valid optimizer type!")

    if args.mode == "test" and args.checkpoint_path != "":
        print("loading model from checkpoint...")
        load_model(args, model, optimizer)
    model = model.to(args.device)

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    # Main Function
    if args.mode == "train":
        stat_dict = {"train/loss": []}
        for epoch in range(args.num_epoch, args.max_epoch):
            train_one_epoch(epoch, loader, args, model, criterion, optimizer, stat_dict)

            if epoch % args.val_freq == 0:
                evaluate_one_epoch(val_loader, args, model, criterion="acc")

            if epoch % args.save_freq == 0:
                save_model(args, model, optimizer, epoch)

        save_model(args, model, optimizer)
        print("[Main] Model training has been completed!")

    elif args.mode == "test":
        print("[Main] Model testing...")
        evaluate_one_epoch(loader, args, model, criterion=None, save_name="result.txt")

    else:
        raise NotImplementedError("You must specify either to train or to test!")
