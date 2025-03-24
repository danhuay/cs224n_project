import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def get_tsa_threshold(schedule, global_step, total_steps, n_classes):
    training_progress = global_step / total_steps
    start = 1 / n_classes  # equal probability for each class
    end = 1
    scale = 5  # empirical scale

    if schedule == "linear":
        threshold = training_progress
    elif schedule == "exp":
        threshold = math.exp((training_progress - 1) * scale)
    elif schedule == "log":
        threshold = 1 - math.exp((-training_progress) * scale)
    elif schedule == "constant":
        threshold = 1
    else:
        raise ValueError("Unknown schedule")
    return threshold * (end - start) + start


def training_signal_annealing(
    logits, y_target, global_step, total_steps, schedule="linear_schedule"
):
    """
    logits is the output from the model in shape of (batch_size, n_class)
    y_target in shape of (batch_size, )
    threshold_func can be a partial that only takes in global step as argument
    """

    # get tsa threshold
    one_hot_label = F.one_hot(y_target, num_classes=logits.size(1))
    n_classes = one_hot_label.size(1)
    threshold = get_tsa_threshold(schedule, global_step, total_steps, n_classes)

    # create sample filtering mask
    y_pred_prob = F.softmax(logits, dim=1)
    correct_answer_confidence = torch.sum(y_pred_prob * one_hot_label, dim=1)
    mask = (correct_answer_confidence <= threshold).float()
    mask.requires_grad = False

    per_sample_loss = F.cross_entropy(logits, y_target, reduction="none")

    # Summing the per-example losses
    total_loss = torch.sum(per_sample_loss * mask)

    # Summing the mask values and ensuring the denominator is at least 1
    total_mask = torch.maximum(torch.sum(mask), torch.tensor(1.0))

    # Calculating the supervised loss as the mean of the masked per-example losses
    loss = total_loss / total_mask
    return loss


def kl_divergence(unsup_prob, aug_prob):

    log_p = torch.log(unsup_prob)
    q = aug_prob
    # Calculate KL divergence using kl_div
    kl = F.kl_div(log_p, q, reduction="none", log_target=False)
    # Sum across the classes (assuming log_p and log_q are shape [batch_size, num_classes])
    kl = kl.sum(dim=-1)
    return kl


def confidence_masking(logits_unsup, logits_aug, temperature=0.4, beta=0.8):
    # temperature for sharpening
    # unsup part stops gradient
    # use temperature to sharpen the prediction
    y_pred_unsup_prob_sharpen = F.softmax(
        logits_unsup.detach() / temperature, dim=1
    ).detach()
    y_pred_unsup_prob = F.softmax(logits_unsup.detach(), dim=1).detach()
    y_pred_aug_prob = F.softmax(logits_aug, dim=1)

    # confidence masking
    y_pred_max_values = torch.max(y_pred_unsup_prob, dim=-1).values
    mask = (y_pred_max_values > beta).float()
    mask.requires_grad = False

    # Summing the per-example losses
    per_example_kl_loss = kl_divergence(y_pred_unsup_prob_sharpen, y_pred_aug_prob)
    total_loss = torch.sum(per_example_kl_loss * mask)

    # Summing the mask values and ensuring the denominator is at least 1
    total_mask = torch.maximum(torch.sum(mask), torch.tensor(1.0))

    # Calculating the supervised loss as the mean of the masked per-example losses
    loss = total_loss / total_mask
    return loss


def uda_loss(
    logits,
    y_target,
    global_step,
    total_steps,
    schedule,
    lmbd=1,
    beta=0.8,
    temperature=0.4,
):
    # the index where predictions become -1
    unsup_index_start = torch.sum(y_target >= 0)
    aug_index_start = (len(y_target) + unsup_index_start) // 2

    logits_sup = logits[:unsup_index_start]
    logits_unsup = logits[unsup_index_start:aug_index_start]
    logits_aug = logits[aug_index_start:]

    y_target = y_target[:unsup_index_start]

    # supervised loss
    sup_loss = training_signal_annealing(
        logits_sup, y_target, global_step, total_steps, schedule
    )

    # unsupervised loss, is per sample loss
    unsup_loss = confidence_masking(logits_unsup, logits_aug, temperature, beta=beta)

    if sup_loss is not None and unsup_loss is not None:
        loss = sup_loss + lmbd * unsup_loss
    elif sup_loss is not None:
        loss = sup_loss
    elif unsup_loss is not None:
        loss = unsup_loss
    else:
        loss = None
    return loss
