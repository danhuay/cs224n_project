import torch
import torch.nn as nn
import random
import numpy as np
from collections import OrderedDict
import click

from src.trainer import Trainer
from src.multitask_bert import MultitaskBERT
from src.optimizer import AdamW
from src.datasets import get_split_data_loaders, get_split_unsup_aug_data_loaders
from src.utils import load_yaml_to_simplenamespace, convert_number
from src.loss import training_signal_annealing, uda_loss

import sys
import logging

# logging setup
# Set up logger
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    # filemode="w+",vim
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


@click.command()
@click.option("--debug", is_flag=True, help="Debug mode")
@click.option("--name", default="baseline", help="Experiment name")
@click.option("--params", multiple=True, help="Parameters to override")
def main(debug, name, params):
    # loading configs
    config = load_yaml_to_simplenamespace("multitask_config.yml")
    device = torch.device(
        "cuda" if torch.cuda.is_available() and config.use_gpu else "cpu"
    )

    # override config with command line arguments
    params_dict = {}
    for param in params:
        key, values = param.split("=")
        if "," in values:
            values_list = [convert_number(val) for val in values.split(",")]
            values_list = [
                val for val in values_list if len(val) > 0
            ]  # remove empty strings
        else:
            values_list = convert_number(values)
        params_dict[key] = values_list

    # update config with command line arguments
    for key, values in params_dict.items():
        setattr(config, key, values)

    # reconstruct model checkpoint path
    model_checkpoint_path = config.model_checkpoint_path.format(
        fine_tune_mode=config.fine_tune_mode,
        epochs=config.epochs,
        lr=config.learning_rate,
    )
    # to avoid the error of "too many open files" when using multiple workers
    if config.cpu_workers > 0:
        torch.multiprocessing.set_sharing_strategy("file_system")

    # seed everything
    seed_everything(config.seed)

    # get data loaders
    if config.supervised:
        train_loader = get_split_data_loaders(
            debug=debug,
            num_workers=config.cpu_workers,
            batch_size=config.batch_size,
            **vars(config.train_data_loader_config),
        )
        loss_fn = training_signal_annealing
    else:
        train_loader = get_split_unsup_aug_data_loaders(
            debug=debug,
            num_workers=config.cpu_workers,
            aug_approach=config.aug_approach,
            **vars(config.train_unsup_data_loader_config),
        )
        loss_fn = uda_loss
    dev_loader = get_split_data_loaders(
        debug=debug,
        num_workers=config.cpu_workers,
        batch_size=config.batch_size,
        **vars(config.dev_data_loader_config),
    )

    # Initialize the model, loss function, and optimizer
    model = MultitaskBERT(config.multitask_model_config, config.fine_tune_mode)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)

    # Initialize TensorBoard writer, None if debugging
    if not debug:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(f"runs/{name}")
    else:
        writer = None

    # this is to select which part of the data to be used as training
    train_loader_sub = OrderedDict()
    for tag in config.train_data_tags:
        train_loader_sub[tag] = train_loader[tag]

    # Train the model with multiple tasks
    trainer = Trainer(
        model,
        train_loader_sub,
        dev_loader,
        optimizer,
        device=device,
        epochs=config.epochs,
        writer=writer,
        model_checkpoint_path=model_checkpoint_path,
        config=config,
        loss_fn=loss_fn,
    )

    # Train the model
    if config.mode == "train":
        trainer.train(config.training_approach)
        # load the best performing model
        trainer.load_model(model_checkpoint_path)

    elif config.mode == "inference":
        trainer.load_model(config.model_checkpoint_path_inference)

        # dev_res = trainer.evaluate_model(
        #     trainer.dev_dataloader["sst"],
        #     trainer.dev_dataloader["para"],
        #     trainer.dev_dataloader["sts"],
        # )
        #
        # logger.info(f"Dev SST results: {dev_res[0]:.3f}")
        # logger.info(f"Dev Para results: {dev_res[1]:.3f}")
        # logger.info(f"Dev STS results: {dev_res[2]:.3f}")
        # logger.info(f"Dev Pooled results: {dev_res[3]:.3f}")
        #
        # trainer.save_predictions(*dev_res[-1], vars(config.dev_outputs))

    if config.need_testing:
        test_loader = get_split_data_loaders(
            num_workers=config.cpu_workers,
            batch_size=config.batch_size,
            **vars(config.test_data_loader_config),
        )

        result_group = trainer.test_model(
            test_loader["sst"],
            test_loader["para"],
            test_loader["sts"],
        )

        trainer.save_predictions(*result_group, vars(config.test_outputs))

    # Close the writer
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
