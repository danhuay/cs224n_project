#!/usr/bin/env python3

"""
Multitask BERT evaluation functions.

When training your multitask model, you will find it useful to call
model_eval_multitask to evaluate your model on the 3 tasks' dev sets.
"""

import torch
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
import numpy as np
import logging

logger = logging.getLogger(__name__)

TQDM_DISABLE = False


# Evaluate multitask model on SST only.
# def model_eval_sst(dataloader, model, device, early_stop_step=None):
#     model.eval()  # Switch to eval model, will turn off randomness like dropout.
#     y_true = []
#     y_pred = []
#     sents = []
#     sent_ids = []
#     for step, batch in enumerate(tqdm(dataloader, desc=f"eval", disable=TQDM_DISABLE)):
#         if early_stop_step is not None and step > early_stop_step:
#             break
#         b_ids, b_mask, b_labels, b_sents, b_sent_ids = (
#             batch["token_ids"],
#             batch["attention_mask"],
#             batch["labels"],
#             batch["sents"],
#             batch["sent_ids"],
#         )
#
#         b_ids = b_ids.to(device)
#         b_mask = b_mask.to(device)
#
#         logits = model.predict_sentiment(b_ids, b_mask)
#         logits = logits.detach().cpu().numpy()
#         preds = np.argmax(logits, axis=1).flatten()
#
#         b_labels = b_labels.flatten()
#         y_true.extend(b_labels)
#         y_pred.extend(preds)
#         sents.extend(b_sents)
#         sent_ids.extend(b_sent_ids)
#
#     f1 = f1_score(y_true, y_pred, average="macro")
#     acc = accuracy_score(y_true, y_pred)
#
#     return acc, f1, y_pred, y_true, sents, sent_ids


# Evaluate multitask model on dev sets.
def model_eval_sst(sentiment_dataloader, model, device, early_stop_step=None):
    model.eval()
    with torch.no_grad():
        sst_y_true = []
        sst_y_pred = []
        sst_sent_ids = []
        for step, batch in enumerate(
            tqdm(sentiment_dataloader, desc=f"eval", disable=TQDM_DISABLE)
        ):
            if early_stop_step is not None and step > early_stop_step:
                break
            b_ids, b_mask, b_labels, b_sent_ids = (
                batch["token_ids"],
                batch["attention_mask"],
                batch["labels"],
                batch["sent_ids"],
            )

            non_minus_1_mask = b_labels != -1
            minus_1_mask = b_labels == -1
            if minus_1_mask.all():
                continue

            b_ids = b_ids[non_minus_1_mask].to(device)
            b_mask = b_mask[non_minus_1_mask].to(device)
            b_labels = b_labels[non_minus_1_mask]

            logits = model.predict_sentiment(b_ids, b_mask)
            y_hat = logits.argmax(dim=-1).flatten().cpu().numpy()
            b_labels = b_labels.flatten().cpu().numpy()

            sst_y_pred.extend(y_hat)
            sst_y_true.extend(b_labels)
            sst_sent_ids.extend(b_sent_ids)

        sentiment_accuracy = np.mean(np.array(sst_y_pred) == np.array(sst_y_true))
        return sentiment_accuracy, sst_y_pred, sst_sent_ids


def model_eval_para(paraphrase_dataloader, model, device, early_stop_step=None):
    model.eval()
    with torch.no_grad():
        para_y_true = []
        para_y_pred = []
        para_sent_ids = []
        for step, batch in enumerate(
            tqdm(paraphrase_dataloader, desc=f"eval", disable=TQDM_DISABLE)
        ):
            if early_stop_step is not None and step > early_stop_step:
                break
            (b_ids1, b_mask1, b_ids2, b_mask2, b_labels, b_sent_ids) = (
                batch["token_ids_1"],
                batch["attention_mask_1"],
                batch["token_ids_2"],
                batch["attention_mask_2"],
                batch["labels"],
                batch["sent_ids"],
            )

            non_minus_1_mask = b_labels != -1
            minus_1_mask = b_labels == -1
            if minus_1_mask.all():
                continue

            b_ids1 = b_ids1[non_minus_1_mask].to(device)
            b_mask1 = b_mask1[non_minus_1_mask].to(device)
            b_ids2 = b_ids2[non_minus_1_mask].to(device)
            b_mask2 = b_mask2[non_minus_1_mask].to(device)
            b_labels = b_labels[non_minus_1_mask]

            logits = model.predict_paraphrase(b_ids1, b_mask1, b_ids2, b_mask2)
            # y_hat = logits.sigmoid().round().flatten().cpu().numpy()
            y_hat = logits.argmax(dim=-1).flatten().cpu().numpy()
            b_labels = b_labels.flatten().cpu().numpy()

            para_y_pred.extend(y_hat)
            para_y_true.extend(b_labels)
            para_sent_ids.extend(b_sent_ids)

        paraphrase_accuracy = np.mean(np.array(para_y_pred) == np.array(para_y_true))
        return paraphrase_accuracy, para_y_pred, para_sent_ids


def model_eval_sts(sts_dataloader, model, device, early_stop_step=None):
    model.eval()
    with torch.no_grad():
        sts_y_true = []
        sts_y_pred = []
        sts_sent_ids = []
        for step, batch in enumerate(
            tqdm(sts_dataloader, desc=f"eval", disable=TQDM_DISABLE)
        ):
            if early_stop_step is not None and step > early_stop_step:
                break
            (b_ids1, b_mask1, b_ids2, b_mask2, b_labels, b_sent_ids) = (
                batch["token_ids_1"],
                batch["attention_mask_1"],
                batch["token_ids_2"],
                batch["attention_mask_2"],
                batch["labels"],
                batch["sent_ids"],
            )
            non_minus_1_mask = b_labels != -1
            minus_1_mask = b_labels == -1
            if minus_1_mask.all():
                continue

            b_ids1 = b_ids1[non_minus_1_mask].to(device)
            b_mask1 = b_mask1[non_minus_1_mask].to(device)
            b_ids2 = b_ids2[non_minus_1_mask].to(device)
            b_mask2 = b_mask2[non_minus_1_mask].to(device)
            b_labels = b_labels[non_minus_1_mask]

            logits = model.predict_similarity(b_ids1, b_mask1, b_ids2, b_mask2)
            # y_hat = logits.flatten().cpu().numpy()
            y_hat = logits.argmax(dim=-1).flatten().cpu().numpy()
            b_labels = b_labels.flatten().cpu().numpy()

            sts_y_pred.extend(y_hat * 0.2)  # convert to 0-5 scale
            sts_y_true.extend(b_labels * 0.2)  # convert to 0-5 scale
            sts_sent_ids.extend(b_sent_ids)
        pearson_mat = np.corrcoef(sts_y_pred, sts_y_true)
        sts_corr = pearson_mat[1][0]
        return sts_corr, sts_y_pred, sts_sent_ids


def model_eval_multitask(
    sentiment_dataloader,
    paraphrase_dataloader,
    sts_dataloader,
    model,
    device,
    early_stop_step=None,
):
    sentiment_accuracy, sst_y_pred, sst_sent_ids = model_eval_sst(
        sentiment_dataloader, model, device, early_stop_step
    )
    paraphrase_accuracy, para_y_pred, para_sent_ids = model_eval_para(
        paraphrase_dataloader, model, device, early_stop_step
    )
    sts_corr, sts_y_pred, sts_sent_ids = model_eval_sts(
        sts_dataloader, model, device, early_stop_step
    )

    return (
        sentiment_accuracy,
        sst_y_pred,
        sst_sent_ids,
        paraphrase_accuracy,
        para_y_pred,
        para_sent_ids,
        sts_corr,
        sts_y_pred,
        sts_sent_ids,
    )


# Evaluate multitask model on test sets.
def model_eval_test_multitask(
    sentiment_dataloader, paraphrase_dataloader, sts_dataloader, model, device
):
    model.eval()  # Switch to eval model, will turn off randomness like dropout.

    with torch.no_grad():
        # Evaluate sentiment classification.
        sst_y_pred = []
        sst_sent_ids = []
        for step, batch in enumerate(
            tqdm(sentiment_dataloader, desc=f"eval", disable=TQDM_DISABLE)
        ):
            b_ids, b_mask, b_sent_ids = (
                batch["token_ids"],
                batch["attention_mask"],
                batch["sent_ids"],
            )

            b_ids = b_ids.to(device)
            b_mask = b_mask.to(device)

            logits = model.predict_sentiment(b_ids, b_mask)
            y_hat = logits.argmax(dim=-1).flatten().cpu().numpy()

            sst_y_pred.extend(y_hat)
            sst_sent_ids.extend(b_sent_ids)

        # Evaluate paraphrase detection.
        para_y_pred = []
        para_sent_ids = []
        for step, batch in enumerate(
            tqdm(paraphrase_dataloader, desc=f"eval", disable=TQDM_DISABLE)
        ):
            (b_ids1, b_mask1, b_ids2, b_mask2, b_sent_ids) = (
                batch["token_ids_1"],
                batch["attention_mask_1"],
                batch["token_ids_2"],
                batch["attention_mask_2"],
                batch["sent_ids"],
            )

            b_ids1 = b_ids1.to(device)
            b_mask1 = b_mask1.to(device)
            b_ids2 = b_ids2.to(device)
            b_mask2 = b_mask2.to(device)

            logits = model.predict_paraphrase(b_ids1, b_mask1, b_ids2, b_mask2)
            y_hat = logits.argmax(dim=-1).flatten().cpu().numpy()

            para_y_pred.extend(y_hat)
            para_sent_ids.extend(b_sent_ids)

        # Evaluate semantic textual similarity.
        sts_y_pred = []
        sts_sent_ids = []
        for step, batch in enumerate(
            tqdm(sts_dataloader, desc=f"eval", disable=TQDM_DISABLE)
        ):
            (b_ids1, b_mask1, b_ids2, b_mask2, b_sent_ids) = (
                batch["token_ids_1"],
                batch["attention_mask_1"],
                batch["token_ids_2"],
                batch["attention_mask_2"],
                batch["sent_ids"],
            )

            b_ids1 = b_ids1.to(device)
            b_mask1 = b_mask1.to(device)
            b_ids2 = b_ids2.to(device)
            b_mask2 = b_mask2.to(device)

            logits = model.predict_similarity(b_ids1, b_mask1, b_ids2, b_mask2)
            y_hat = logits.argmax(dim=-1).flatten().cpu().numpy()

            sts_y_pred.extend(y_hat * 0.2)
            sts_sent_ids.extend(b_sent_ids)

        return (
            sst_y_pred,
            sst_sent_ids,
            para_y_pred,
            para_sent_ids,
            sts_y_pred,
            sts_sent_ids,
        )
