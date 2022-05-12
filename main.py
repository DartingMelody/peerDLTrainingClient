# coding=utf-8
# Copyright (c) Microsoft Corporation. Licensed under the MIT license.
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning multi-lingual models on XNLI (Bert, DistilBERT, XLM).
    Adapted from `examples/run_glue.py`"""


import argparse
from glob import glob
import logging
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
import time

from transformers import (
    AdamW,
    DistilBertConfig, DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)

logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def load_and_cache_examples(vocab_size, num_labels, seq_length, num_examples):
    all_input_ids = torch.tensor([[random.randint(0, vocab_size - 1) for __ in range(seq_length)] for _ in range(num_examples)], dtype=torch.long)
    all_attention_mask = torch.tensor([[random.randint(0, 1) for __ in range(seq_length)] for _ in range(num_examples)], dtype=torch.long)
    all_labels = torch.tensor([random.randint(0, num_labels - 1) for _ in range(num_examples)], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_labels)

    return dataset


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name_or_path",
        default="config.json",
        type=str,
        # required=True,
        help="Path to pre-trained model or shortcut name selected in the list: ",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--checkpoint_frequency", default=1024, type=int)
    parser.add_argument("--per_gpu_batch_size", default=32, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument(
        "--num_train_epochs", default=10.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--num_examples", type=int, default=32678, help="Number of examples to simulate")
    parser.add_argument("--seq_length", type=int, default=128, help="Sequence length of simulated examples")
    parser.add_argument("--num_labels", type=int, default=2, help="Number of labels for classification task")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    args = parser.parse_args()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
    )

    # Set seed
    set_seed(args)

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
    ):
        listing = os.listdir(args.output_dir)
        try:
            listing.remove('progress')
        except:
            pass
        if len(listing) > 0:
            last_checkpoint = sorted(listing)[-1]
            global_step = int(last_checkpoint)
            model = DistilBertForSequenceClassification.from_pretrained(args.output_dir + '/' + last_checkpoint)
        else:
            config = DistilBertConfig.from_pretrained(args.model_name_or_path, num_labels=args.num_labels)
            model = DistilBertForSequenceClassification(config=config)
            global_step = 0    
    else:
        config = DistilBertConfig.from_pretrained(args.model_name_or_path, num_labels=args.num_labels)
        model = DistilBertForSequenceClassification(config=config)
        global_step = 0

    model.to(args.device)

    train_dataset = load_and_cache_examples(model.config.vocab_size, args.num_labels, args.seq_length, args.num_examples)
    args.train_batch_size = args.per_gpu_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=t_total)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    epochs_trained = global_step // len(train_dataloader)
    steps_trained_in_current_epoch = global_step % len(train_dataloader)

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    set_seed(args)  # Added here for reproductibility
    if (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        progress_file = open(args.output_dir+'/progress', 'a')
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[2]}
            outputs = model(**inputs, return_dict=False)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            tr_loss += loss.item()
            global_step += 1
            if (args.local_rank == -1 or torch.distributed.get_rank() == 0):
                progress_file.write(str(global_step)+ "," + str(time.time()) + '\n')
                progress_file.flush()
            if (global_step) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
            if global_step % args.checkpoint_frequency == 0:
                if (args.local_rank == -1 or torch.distributed.get_rank() == 0):
                    if args.output_dir is not None:
                        logger.info("Saving checkpoint at step {}".format(global_step))
                        model_to_save = model.module if hasattr(model, 'module') else model
                        model_to_save.save_pretrained(args.output_dir + '/' + str(global_step))
    if (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        progress_file.close()

if __name__ == "__main__":
    main()
