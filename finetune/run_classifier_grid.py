"""
This script provides an example to wrap UER-py for classification with grid search.
"""
import sys
import os
import torch
import random
import argparse
from itertools import product

uer_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(uer_dir)

from uer.opts import *
from finetune.run_classifier import *


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    finetune_opts(parser)

    parser.add_argument("--pooling", choices=["mean", "max", "first", "last"], default="first",
                        help="Pooling type.")

    tokenizer_opts(parser)

    parser.add_argument("--soft_targets", action='store_true',
                        help="Train model with logits.")
    parser.add_argument("--soft_alpha", type=float, default=0.5,
                        help="Weight of the soft targets loss.")

    parser.add_argument("--batch_size_list", default=[32, 64], nargs='+', type=int,
                        help="A list of batch sizes for grid search.")
    parser.add_argument("--learning_rate_list", type=float, default=[3e-5, 1e-4, 3e-4], nargs='+',
                        help="A list of learning rate for grid search.")
    parser.add_argument("--epochs_num_list", type=int, default=[3, 5, 8], nargs='+',
                        help="A list of number of epochs for grid search.")

    adv_opts(parser)

    args = parser.parse_args()
    #print("grid start1", args.emb_size)

    # Load the hyperparameters from the config file.
    args = load_hyperparam(args)
    #print("grid start2", args.emb_size)

    set_seed(args.seed)

    # Count the number of labels.
    args.labels_num = count_labels_num(args.train_path)

    #print("grid start3", args.emb_size)
    # Build tokenizer.
    args.tokenizer = str2tokenizer[args.tokenizer](args)
    #print("grid start4", args.emb_size)

    best_acc = 0
    config = {}

    #Build dataset
    trainset = read_dataset(args, args.train_path)
    random.shuffle(trainset)
    instances_num = len(trainset)

    src = torch.LongTensor([example[0] for example in trainset])
    tgt = torch.LongTensor([example[1] for example in trainset])
    seg = torch.LongTensor([example[2] for example in trainset])
    if args.soft_targets:
        soft_tgt = torch.FloatTensor([example[3] for example in trainset])
    else:
        soft_tgt = None

    #print("grid start5", args.emb_size)
    for batch_size, learning_rate, epochs_num in product(args.batch_size_list, args.learning_rate_list, args.epochs_num_list):
        print(batch_size, learning_rate, epochs_num)

        #print("grid start6", args.emb_size)
        args.learning_rate = learning_rate
        args.batch_size = batch_size
        args.epochs_num = epochs_num

        args.train_steps = int(instances_num * args.epochs_num / batch_size) + 1

        # Build classification model.
        #print("grid", args.emb_size)
        model = Classifier(args)

        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(args.device)
        load_or_initialize_parameters(args, model)
        optimizer, scheduler = build_optimizer(args, model)
        if args.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, optimizer = amp.initialize(model, optimizer, opt_level = args.fp16_opt_level)
            args.amp = amp
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        args.model = model

        if args.use_adv:
            args.adv_method = str2adv[args.adv_type](model)

        # Training phase.

        total_loss, _, _ = 0., 0., 0.
        global_step = 0
        args.report_steps = 100
        for epoch in range(1, args.epochs_num+1):
            model.train()
            for i, (src_batch, tgt_batch, seg_batch, soft_tgt_batch) in enumerate(batch_loader(batch_size, src, tgt, seg, soft_tgt)):
                global_step += 1
                _ = train_model(args, model, optimizer, scheduler, src_batch, tgt_batch, seg_batch, soft_tgt_batch)
                if global_step % args.report_steps == 0:
                    acc, _ = evaluate(args, read_dataset(args, args.dev_path))
                    if acc > best_acc:
                        best_acc = acc
                        config = {"learning_rate": learning_rate, "batch_size": batch_size, "epochs_num": epochs_num}
                    print("Epoch id: {}, Training steps: {}, best acc: {:.3f}".format(epoch, global_step, best_acc))
    print("Best Acc. is: {:.4f}, on configuration {}.".format(best_acc, config))

if __name__ == "__main__":
    main()
