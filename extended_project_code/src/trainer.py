import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import random
import numpy as np
from tqdm import tqdm
from itertools import zip_longest
from typing import *
from functools import reduce

from .evaluation import *
from .utils import create_folder_if_not_exists
from .loss import uda_loss, training_signal_annealing
import logging

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        model,
        dataloader: Dict[str, DataLoader],
        dev_dataloader: Dict[str, DataLoader],
        optimizer,
        device,
        epochs,
        writer,
        model_checkpoint_path,
        config,
        loss_fn=None,
    ):
        self.model = model.to(device)
        self.dataloader = dataloader
        self.dev_dataloader = dev_dataloader
        self.optimizer = optimizer
        self.device = device
        self.epochs = epochs
        self.writer = writer
        self.model_checkpoint_path = model_checkpoint_path
        self.config = config
        self.is_trained = False
        self.training_funcs = {
            "sst": self.train_sst_batch,
            "para": self.train_para_batch,
            "sts": self.train_sts_batch,
        }
        self.eval_funcs = {
            "sst": model_eval_sst,
            "para": model_eval_para,
            "sts": model_eval_sts,
        }
        self.global_step = 0  # initialize global step
        self.patience = 0  # initialize patience for early stopping
        self.early_stop_patience = config.early_stop_patience
        self.loss_fn = loss_fn

    def train_sst_batch(self, batch, total_steps=None):
        b_ids, b_mask, b_labels = (
            batch["token_ids"].to(self.device),
            batch["attention_mask"].to(self.device),
            batch["labels"].to(self.device),
        )

        logits = self.model.predict_sentiment(b_ids, b_mask)

        if total_steps is None:
            total_steps = self.epochs * len(self.dataloader["sst"])

        loss = self.loss_fn(
            logits=logits,
            y_target=b_labels,
            global_step=self.global_step,
            total_steps=total_steps,
            schedule=self.config.tsa_schedule,
        )
        return loss

    def train_para_batch(self, batch, total_steps=None):
        b_ids1, b_mask1, b_ids2, b_mask2, b_labels = (
            batch["token_ids_1"].to(self.device),
            batch["attention_mask_1"].to(self.device),
            batch["token_ids_2"].to(self.device),
            batch["attention_mask_2"].to(self.device),
            batch["labels"].to(self.device),
        )

        logits = self.model.predict_paraphrase(b_ids1, b_mask1, b_ids2, b_mask2)
        # loss = nn.BCEWithLogitsLoss()(logits.squeeze(-1), b_labels.float().view(-1))
        if total_steps is None:
            total_steps = self.epochs * len(self.dataloader["para"])

        loss = self.loss_fn(
            logits=logits,
            y_target=b_labels,
            global_step=self.global_step,
            total_steps=total_steps,
            schedule=self.config.tsa_schedule,
        )
        return loss

    def train_sts_batch(self, batch, total_steps=None):
        b_ids1, b_mask1, b_ids2, b_mask2, b_labels = (
            batch["token_ids_1"].to(self.device),
            batch["attention_mask_1"].to(self.device),
            batch["token_ids_2"].to(self.device),
            batch["attention_mask_2"].to(self.device),
            batch["labels"].to(self.device),
        )

        logits = self.model.predict_similarity(b_ids1, b_mask1, b_ids2, b_mask2)
        # loss = nn.MSELoss()(logits.squeeze(-1), b_labels.view(-1))
        if total_steps is None:
            total_steps = self.epochs * len(self.dataloader["sts"])

        loss = self.loss_fn(
            logits=logits,
            y_target=b_labels,
            global_step=self.global_step,
            total_steps=total_steps,
            schedule=self.config.tsa_schedule,
        )
        return loss

    def simultaneous_train_epoch_curriculum(self, epoch, exhaustive=True):
        # train different tasks in the same batch, and return a combined loss
        # if exhaustive=True, will exhaust the data loader have the most batches
        # if exhaustive=False, will stop when the shortest data loader is exhausted
        if exhaustive:
            training_batches = list(zip_longest(*self.dataloader.values()))
        else:
            training_batches = list(zip(*self.dataloader.values()))
        total_steps = len(training_batches) * self.epochs
        data_order = list(self.dataloader.keys())
        training_funcs = [self.training_funcs[key] for key in data_order]

        self.model.train()
        train_loss = 0  # training loss for the epoch
        num_batches = 0
        for batch in tqdm(
            training_batches, desc=f"train-combined-{epoch}", disable=False
        ):
            num_batches += 1
            self.global_step += 1
            self.optimizer.zero_grad()

            # compute loss for each task separately
            losses = {}
            for key_i, pair in enumerate(zip(training_funcs, batch)):
                training_func, task_batch = pair
                if task_batch is not None:
                    loss = training_func(task_batch, total_steps=total_steps)
                    losses[data_order[key_i]] = loss

            # TODO: Combine the losses in different way
            _losses = [val for val in losses.values() if val is not None]
            if len(_losses) == 0:
                continue

            pooled_loss = reduce(lambda x, y: x + y, _losses)

            pooled_loss.backward()
            
            nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
            
            self.optimizer.step()

            # plotted train_loss
            train_loss += pooled_loss.item()

            if self.writer is not None:
                current_step_avg_train_loss = train_loss / num_batches
                if num_batches % 5 == 0:
                    self.writer.add_scalar(
                        f"Loss/combined-train",
                        current_step_avg_train_loss,
                        self.global_step,
                    )

        epoch_avg_train_loss = train_loss / num_batches

        # getting pooled performance
        (
            sentiment_accuracy,
            paraphrase_accuracy,
            sts_corr,
            train_acc,
            result_group,
        ) = self.evaluate_model(
            self.dataloader["sst"],
            self.dataloader["para"],
            self.dataloader["sts"],
        )

        if self.writer is not None:
            self.writer.add_scalar(f"_TrainEval/combined-train", train_acc, epoch)
        logger.info(
            f"Epoch {epoch}: train loss :: {epoch_avg_train_loss:.3f}, "
            f"train acc :: {train_acc:.3f}"
        )

        return

    def sequential_train_epoch_curriculum(self, epoch):
        self.model.train()
        for key, task_data_loader in self.dataloader.items():
            logger.info(f"Training on {key} dataset")
            training_eval_func = self.eval_funcs[key]
            train_loss = 0
            num_batches = 0
            for batch in tqdm(
                task_data_loader, desc=f"train-{key}-{epoch}", disable=False
            ):
                self.global_step += 1

                self.optimizer.zero_grad()

                training_func = self.training_funcs[key]
                loss = training_func(batch)
                if loss is None:
                    continue  # for tsa, if the batch is empty, move on

                loss.backward()
                self.optimizer.step()

                # only update num_batches when this batch is not empty
                num_batches += 1
                # plotted train_loss
                train_loss += loss.item()
                if self.writer is not None:
                    current_step_avg_train_loss = train_loss / num_batches
                    if num_batches % 5 == 0:
                        self.writer.add_scalar(
                            f"Loss/{key}-train",
                            current_step_avg_train_loss,
                            self.global_step,
                        )

            epoch_avg_train_loss = train_loss / num_batches
            train_acc, *_ = training_eval_func(
                task_data_loader,
                self.model,
                self.device,
            )

            if self.writer is not None:
                self.writer.add_scalar(f"_TrainEval/{key}-train", train_acc, epoch)
            logger.info(
                f"Epoch {epoch}: train loss :: {epoch_avg_train_loss:.3f}, "
                f"train acc :: {train_acc:.3f}"
            )

    def train(self, approach="sequential"):
        self.model.train()
        best_dev_acc = 0
        for epoch in range(self.epochs):
            logger.info(f"Epoch {epoch}: Training...")
            if approach == "sequential":
                self.sequential_train_epoch_curriculum(epoch)
            elif approach == "simultaneous":
                self.simultaneous_train_epoch_curriculum(epoch)
            else:
                raise ValueError(f"Unknown training approach: {approach}")

            logger.info(f"Epoch {epoch}: Evaluating on dev sets...")

            (
                sentiment_accuracy,
                paraphrase_accuracy,
                sts_corr,
                pooled_score,
                result_group,
            ) = self.evaluate_model(
                self.dev_dataloader["sst"],
                self.dev_dataloader["para"],
                self.dev_dataloader["sts"],
            )

            logger.info("=" * 50)
            logger.info(f"Epoch {epoch}: Dev evaluation results:")
            logger.info(f"sst | para | sts | pooled_score")
            logger.info(
                f"{sentiment_accuracy:.3f} | "
                f"{paraphrase_accuracy:.3f} | "
                f"{sts_corr:.3f} | "
                f"{pooled_score:.3f}"
            )
            logger.info("=" * 50)

            if self.writer is not None:
                self.writer.add_scalar(
                    "_DevEval/sst_accuracy", sentiment_accuracy, epoch
                )
                self.writer.add_scalar(
                    "_DevEval/para_accuracy", paraphrase_accuracy, epoch
                )
                self.writer.add_scalar("_DevEval/sts_corr", sts_corr, epoch)
                self.writer.add_scalar("_DevEval/pooled_score", pooled_score, epoch)

            dev_acc = pooled_score
            if dev_acc > best_dev_acc:
                logger.info(f"New best dev accuracy: {dev_acc:.3f}")
                best_dev_acc = dev_acc
                self.save_model()
                self.is_trained = True
                self.patience = 0  # reset patience
                # save the best dev predictions
                self.save_predictions(
                    *result_group, files_dir=vars(self.config.dev_outputs)
                )
            else:
                # simple early stopping
                if self.patience >= self.early_stop_patience:
                    logger.warning(f"Early stopping at epoch {epoch}.")
                    break
                else:
                    self.patience += 1
                    logger.warning(
                        f"Model is not improving on dev set. "
                        f"Epochs left: {self.early_stop_patience - self.patience}"
                    )

    @staticmethod
    def pooled_leaderboard_score(sst_acc, para_acc, sts_corr):
        corrected_sts = (sts_corr + 1) / 2
        pooled_score = (sst_acc + para_acc + corrected_sts) / 3
        return pooled_score

    def evaluate_model(
        self,
        sst_data_loader,
        para_data_loader,
        sts_data_loader,
    ):
        # evaluate on multiple tasks
        (
            sentiment_accuracy,
            sst_y_pred,
            sst_sent_ids,
            paraphrase_accuracy,
            para_y_pred,
            para_sent_ids,
            sts_corr,
            sts_y_pred,
            sts_sent_ids,
        ) = model_eval_multitask(
            sst_data_loader,
            para_data_loader,
            sts_data_loader,
            self.model,
            self.device,
        )
        pooled_score = self.pooled_leaderboard_score(
            sentiment_accuracy, paraphrase_accuracy, sts_corr
        )

        result_group = (
            sst_y_pred,
            sst_sent_ids,
            para_y_pred,
            para_sent_ids,
            sts_y_pred,
            sts_sent_ids,
        )

        return (
            sentiment_accuracy,
            paraphrase_accuracy,
            sts_corr,
            pooled_score,
            result_group,
        )

    def test_model(
        self,
        sst_test_dataloader,
        para_test_dataloader,
        sts_test_dataloader,
    ):
        result_group = model_eval_test_multitask(
            sst_test_dataloader,
            para_test_dataloader,
            sts_test_dataloader,
            self.model,
            self.device,
        )

        return result_group

    def save_predictions(
        self,
        sst_y_pred,
        sst_test_sent_ids,
        para_y_pred,
        para_test_sent_ids,
        sts_y_pred,
        sts_test_sent_ids,
        files_dir,
    ):
        """
        files_dir is a dictionary with keys: sst, para, sts and
        values are the path to save the output
        """
        logger.info(f"Saving predictions to {files_dir['sst']}")
        self.saving_output(
            create_folder_if_not_exists(files_dir["sst"]),
            header="id \t Predicted_Sentiment \n",
            y_pred=sst_y_pred,
            sent_ids=sst_test_sent_ids,
        )

        logger.info(f"Saving predictions to {files_dir['para']}")
        self.saving_output(
            create_folder_if_not_exists(files_dir["para"]),
            header="id \t Predicted_Paraphrase \n",
            y_pred=para_y_pred,
            sent_ids=para_test_sent_ids,
        )

        logger.info(f"Saving predictions to {files_dir['sts']}")
        self.saving_output(
            create_folder_if_not_exists(files_dir["sts"]),
            header="id \t Predicted_Similarity \n",
            y_pred=sts_y_pred,
            sent_ids=sts_test_sent_ids,
        )

    @staticmethod
    def saving_output(filename, header, y_pred, sent_ids):
        lines = [f"{sent_id} , {pred} \n" for sent_id, pred in zip(sent_ids, y_pred)]
        with open(filename, "w+") as f:
            f.write(header)
            f.writelines(lines)

    def save_model(self):
        create_folder_if_not_exists(self.model_checkpoint_path)
        torch.save(
            {
                "model": self.model.state_dict(),
                "optim": self.optimizer.state_dict(),
                "system_rng": random.getstate(),
                "numpy_rng": np.random.get_state(),
                "torch_rng": torch.random.get_rng_state(),
            },
            self.model_checkpoint_path,
        )
        logger.info(f"save the model to {self.model_checkpoint_path}")

    def load_model(self, model_checkpoint_path):
        checkpoint = torch.load(model_checkpoint_path)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optim"])
        random.setstate(checkpoint["system_rng"])
        np.random.set_state(checkpoint["numpy_rng"])
        torch.random.set_rng_state(checkpoint["torch_rng"])
        logger.info(f"load the model from {model_checkpoint_path}")
