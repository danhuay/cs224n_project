#!/usr/bin/env python3

"""
This module contains our Dataset classes and functions that load the three datasets
for training and evaluating multitask BERT.

Feel free to edit code in this file if you wish to modify the way in which the data
examples are preprocessed.
"""

import csv
import random
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from .tokenizer import BertTokenizer
from itertools import cycle
import logging

logger = logging.getLogger(__name__)


def preprocess_string(s):
    return " ".join(
        s.lower()
        .replace(".", " .")
        .replace("?", " ?")
        .replace(",", " ,")
        .replace("'", " '")
        .split()
    )


class SentenceClassificationDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def pad_data(self, data):
        sents = [x[0] for x in data]
        labels = [x[1] for x in data]
        sent_ids = [x[2] for x in data]

        encoding = self.tokenizer(
            sents, return_tensors="pt", padding=True, truncation=True
        )
        token_ids = torch.LongTensor(encoding["input_ids"])
        attention_mask = torch.LongTensor(encoding["attention_mask"])
        labels = torch.LongTensor(labels)

        return token_ids, attention_mask, labels, sents, sent_ids

    def collate_fn(self, all_data):
        token_ids, attention_mask, labels, sents, sent_ids = self.pad_data(all_data)

        batched_data = {
            "token_ids": token_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "sents": sents,
            "sent_ids": sent_ids,
        }

        return batched_data


# Unlike SentenceClassificationDataset,
# we do not load labels in SentenceClassificationTestDataset.
# class SentenceClassificationTestDataset(Dataset):
#     def __init__(self, dataset):
#         self.dataset = dataset
#         self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
#
#     def __len__(self):
#         return len(self.dataset)
#
#     def __getitem__(self, idx):
#         return self.dataset[idx]
#
#     def pad_data(self, data):
#         sents = [x[0] for x in data]
#         sent_ids = [x[1] for x in data]
#
#         encoding = self.tokenizer(
#             sents, return_tensors="pt", padding=True, truncation=True
#         )
#         token_ids = torch.LongTensor(encoding["input_ids"])
#         attention_mask = torch.LongTensor(encoding["attention_mask"])
#
#         return token_ids, attention_mask, sents, sent_ids
#
#     def collate_fn(self, all_data):
#         token_ids, attention_mask, sents, sent_ids = self.pad_data(all_data)
#
#         batched_data = {
#             "token_ids": token_ids,
#             "attention_mask": attention_mask,
#             "sents": sents,
#             "sent_ids": sent_ids,
#         }
#
#         return batched_data


class SentencePairDataset(Dataset):
    def __init__(
        self, dataset, isRegression=False, convert_regression_to_classification=False
    ):
        self.dataset = dataset
        self.isRegression = isRegression
        self.convert_regression_to_classification = convert_regression_to_classification
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def pad_data(self, data):
        sent1 = [x[0] for x in data]
        sent2 = [x[1] for x in data]
        labels = [x[2] for x in data]
        sent_ids = [x[3] for x in data]

        encoding1 = self.tokenizer(
            sent1, return_tensors="pt", padding=True, truncation=True
        )
        encoding2 = self.tokenizer(
            sent2, return_tensors="pt", padding=True, truncation=True
        )

        token_ids = torch.LongTensor(encoding1["input_ids"])
        attention_mask = torch.LongTensor(encoding1["attention_mask"])
        token_type_ids = torch.LongTensor(encoding1["token_type_ids"])

        token_ids2 = torch.LongTensor(encoding2["input_ids"])
        attention_mask2 = torch.LongTensor(encoding2["attention_mask"])
        token_type_ids2 = torch.LongTensor(encoding2["token_type_ids"])
        if self.convert_regression_to_classification:
            labels = torch.LongTensor([int(i // 0.2) for i in labels])
        elif self.isRegression:
            labels = torch.FloatTensor(labels)
        else:
            labels = torch.LongTensor(labels)

        return (
            token_ids,
            token_type_ids,
            attention_mask,
            token_ids2,
            token_type_ids2,
            attention_mask2,
            labels,
            sent_ids,
        )

    def collate_fn(self, all_data):
        (
            token_ids,
            token_type_ids,
            attention_mask,
            token_ids2,
            token_type_ids2,
            attention_mask2,
            labels,
            sent_ids,
        ) = self.pad_data(all_data)

        batched_data = {
            "token_ids_1": token_ids,
            "token_type_ids_1": token_type_ids,
            "attention_mask_1": attention_mask,
            "token_ids_2": token_ids2,
            "token_type_ids_2": token_type_ids2,
            "attention_mask_2": attention_mask2,
            "labels": labels,
            "sent_ids": sent_ids,
        }

        return batched_data


class SentencePairDatasetReg2Cls(SentencePairDataset):
    def __init__(self, dataset):
        super().__init__(dataset, convert_regression_to_classification=True)


# Unlike SentencePairDataset, we do not load labels in SentencePairTestDataset.
# class SentencePairTestDataset(Dataset):
#     def __init__(self, dataset, args):
#         self.dataset = dataset
#         self.p = args
#         self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
#
#     def __len__(self):
#         return len(self.dataset)
#
#     def __getitem__(self, idx):
#         return self.dataset[idx]
#
#     def pad_data(self, data):
#         sent1 = [x[0] for x in data]
#         sent2 = [x[1] for x in data]
#         sent_ids = [x[2] for x in data]
#
#         encoding1 = self.tokenizer(
#             sent1, return_tensors="pt", padding=True, truncation=True
#         )
#         encoding2 = self.tokenizer(
#             sent2, return_tensors="pt", padding=True, truncation=True
#         )
#
#         token_ids = torch.LongTensor(encoding1["input_ids"])
#         attention_mask = torch.LongTensor(encoding1["attention_mask"])
#         token_type_ids = torch.LongTensor(encoding1["token_type_ids"])
#
#         token_ids2 = torch.LongTensor(encoding2["input_ids"])
#         attention_mask2 = torch.LongTensor(encoding2["attention_mask"])
#         token_type_ids2 = torch.LongTensor(encoding2["token_type_ids"])
#
#         return (
#             token_ids,
#             token_type_ids,
#             attention_mask,
#             token_ids2,
#             token_type_ids2,
#             attention_mask2,
#             sent_ids,
#         )
#
#     def collate_fn(self, all_data):
#         (
#             token_ids,
#             token_type_ids,
#             attention_mask,
#             token_ids2,
#             token_type_ids2,
#             attention_mask2,
#             sent_ids,
#         ) = self.pad_data(all_data)
#
#         batched_data = {
#             "token_ids_1": token_ids,
#             "token_type_ids_1": token_type_ids,
#             "attention_mask_1": attention_mask,
#             "token_ids_2": token_ids2,
#             "token_type_ids_2": token_type_ids2,
#             "attention_mask_2": attention_mask2,
#             "sent_ids": sent_ids,
#         }
#
#         return batched_data


def read_file(filename, split, process_record):
    data = []
    with open(filename, "r") as fp:
        for record in csv.DictReader(fp, delimiter="\t"):
            data.append(process_record(record))
    logger.info(f"Loaded {len(data)} {split} examples from {filename}")
    return data


def process_sentiment_record(record, emit_label=False):
    sent = record["sentence"].lower().strip()
    sent_id = record["id"].lower().strip()
    if "sentiment" not in record or emit_label:
        label = -1
    else:
        label = int(record["sentiment"].strip())
    return (sent, label, sent_id)


def process_paraphrase_record(record, emit_label=False):
    sent_id = record["id"].lower().strip()
    if "is_duplicate" not in record or emit_label:
        return (
            preprocess_string(record["sentence1"]),
            preprocess_string(record["sentence2"]),
            -1,
            sent_id,
        )
    else:
        return (
            preprocess_string(record["sentence1"]),
            preprocess_string(record["sentence2"]),
            int(float(record["is_duplicate"])),
            sent_id,
        )


def process_similarity_record(record, emit_label=False):
    sent_id = record["id"].lower().strip()
    if "similarity" not in record or emit_label:
        return (
            preprocess_string(record["sentence1"]),
            preprocess_string(record["sentence2"]),
            -1,
            sent_id,
        )
    else:
        return (
            preprocess_string(record["sentence1"]),
            preprocess_string(record["sentence2"]),
            float(record["similarity"]),
            sent_id,
        )


def load_multitask_data(
    sentiment_filename, paraphrase_filename, similarity_filename, split="train"
):
    sentiment_data = read_file(
        sentiment_filename,
        split,
        lambda record: process_sentiment_record(record),
    )
    paraphrase_data = read_file(
        paraphrase_filename,
        split,
        lambda record: process_paraphrase_record(record),
    )
    similarity_data = read_file(
        similarity_filename,
        split,
        lambda record: process_similarity_record(record),
    )

    return sentiment_data, paraphrase_data, similarity_data


def sampler(data, batch_size, frac, is_n_sample=False):
    # sample a fraction of the data that is divisible by the batch size
    if frac == 1:
        return data

    if is_n_sample:
        n_sample = frac
    else:
        n_sample = int(len(data) * frac) // batch_size * batch_size
    if 0 < n_sample < len(data):
        sampled_data = random.sample(data, n_sample)
        return sampled_data
    else:
        logger.warning("Sample fraction is incorrect, returning all data")
        return data


def get_split_data_loaders(
    sst_filename,
    para_filename,
    sts_filename,
    batch_size,
    split="train",
    num_workers=1,
    debug=False,
    sample_frac=None,
):
    # only shuffle the training data
    shuffle = True if split == "train" else False

    # Load the data
    sentiment_data, paraphrase_data, similarity_data = load_multitask_data(
        sst_filename, para_filename, sts_filename, split
    )

    # if debug is True, only use a batch of the data
    if debug:
        sentiment_data = sentiment_data[:batch_size]
        paraphrase_data = paraphrase_data[:batch_size]
        similarity_data = similarity_data[:batch_size]

    # Sampling
    if sample_frac is None:
        sample_frac = {"para": 1.0, "sts": 1.0, "sst": 1.0, "is_n_sample": False}
    elif type(sample_frac) != dict:
        sample_frac = vars(sample_frac)
    sentiment_dataset = SentenceClassificationDataset(
        sampler(
            sentiment_data, batch_size, sample_frac["sst"], sample_frac["is_n_sample"]
        )
    )
    paraphrase_dataset = SentencePairDataset(
        sampler(
            paraphrase_data, batch_size, sample_frac["para"], sample_frac["is_n_sample"]
        )
    )
    similarity_dataset = SentencePairDatasetReg2Cls(
        sampler(
            similarity_data, batch_size, sample_frac["sts"], sample_frac["is_n_sample"]
        )
    )

    logger.info(f"Sampled {split} SST dataset has {len(sentiment_dataset)} rows.")
    logger.info(f"Sampled {split} Para dataset has {len(paraphrase_dataset)} rows.")
    logger.info(f"Sampled {split} STS dataset has {len(similarity_dataset)} rows.")

    # Convert the PyTorch Datasets into DataLoaders
    sentiment_dataloader = DataLoader(
        sentiment_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=sentiment_dataset.collate_fn,
    )
    paraphrase_dataloader = DataLoader(
        paraphrase_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=paraphrase_dataset.collate_fn,
    )
    similarity_dataloader = DataLoader(
        similarity_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=similarity_dataset.collate_fn,
    )

    return {
        "sst": sentiment_dataloader,
        "para": paraphrase_dataloader,
        "sts": similarity_dataloader,
    }


def read_file_unsup_aug(
    filename,
    split,
    process_record,
    aug_approach="back_translation",
):
    """
    aug_approach: values in completion; back_translation; rnd_mask_completion
    unsup_ratio: % of data will be converted into unsupervised data with its corresponding augmentation
    """
    sup_data = dict()
    unsup_data = dict()
    aug_data = dict()

    with open(filename, "r") as fp:
        for record in csv.DictReader(fp, delimiter="\t"):
            # add as supervised data
            _rec = process_record(record, emit_label=False)
            sup_data[_rec[-1]] = _rec

            # add as unsupervised data
            _rec = process_record(record, emit_label=True)
            unsup_data[_rec[-1]] = _rec

    aug_filename = filename[:-4] + "-aug-" + aug_approach + ".csv"
    with open(aug_filename, "r") as fp:
        for record in csv.DictReader(fp, delimiter="\t"):
            _rec = process_record(record, emit_label=True)
            aug_data[_rec[-1]] = _rec

    # make sure the size and order of the two lists are matching
    mutual_keys = list(
        set(unsup_data.keys())
        .intersection(set(aug_data.keys()))
        .intersection(set(sup_data.keys()))
    )
    _sup_data = [sup_data[key] for key in mutual_keys]
    _unsup_data = [unsup_data[key] for key in mutual_keys]
    _aug_data = [aug_data[key] for key in mutual_keys]
    assert len(_unsup_data) == len(_aug_data)

    logger.info(f"Loaded {len(sup_data)} supervised {split} examples from {filename}")
    logger.info(
        f"Loaded {len(unsup_data)} unsupervised {split} examples from {filename}"
    )
    return _sup_data, list(zip(_unsup_data, _aug_data))


def load_multitask_data_with_unsup_aug(
    sentiment_filename,
    paraphrase_filename,
    similarity_filename,
    aug_approach="back_translation",
    split="train",
):
    """
    aug_approach: values in completion; back_translation; rnd_mask_completion
    sup_to_unsup_ratio: sup to unsup ratio
    """
    sentiment_group = read_file_unsup_aug(
        sentiment_filename,
        split,
        process_record=process_sentiment_record,
        aug_approach=aug_approach,
    )
    paraphrase_group = read_file_unsup_aug(
        paraphrase_filename,
        split,
        process_record=process_paraphrase_record,
        aug_approach=aug_approach,
    )
    similarity_group = read_file_unsup_aug(
        similarity_filename,
        split,
        process_record=process_similarity_record,
        aug_approach=aug_approach,
    )

    return sentiment_group, paraphrase_group, similarity_group


def get_split_unsup_aug_data_loaders(
    sst_filename,
    para_filename,
    sts_filename,
    n_sup_batch_size,
    n_unsup_batch_size,
    n_sup_samples,
    n_unsup_samples,
    n_batches=1000,
    split="train",
    aug_approach="back_translation",
    num_workers=1,
    debug=False,
    shuffle=False,
):

    # Load the data
    sentiment_data_group, paraphrase_data_group, similarity_data_group = (
        load_multitask_data_with_unsup_aug(
            sst_filename,
            para_filename,
            sts_filename,
            aug_approach,
            split,
        )
    )

    if debug:
        sentiment_data_group = [
            sentiment_data_group[0][:n_sup_batch_size],
            sentiment_data_group[1][:n_unsup_batch_size],
        ]
        paraphrase_data_group = [
            paraphrase_data_group[0][:n_sup_batch_size],
            paraphrase_data_group[1][:n_unsup_batch_size],
        ]
        similarity_data_group = [
            paraphrase_data_group[0][:n_sup_batch_size],
            paraphrase_data_group[1][:n_unsup_batch_size],
        ]

    # Sampling, must be smaller than the available data size
    def _shuffle_data(data_group, n_sup_samples, n_unsup_samples):
        sup_data, unsup_data = data_group

        if 0 < n_sup_samples < len(sup_data):
            _sup_data = random.sample(sup_data, n_sup_samples)
        else:
            _sup_data = sup_data

        if 0 < n_unsup_samples < len(unsup_data):
            _unsup_data = random.sample(unsup_data, n_unsup_samples)
        else:
            _unsup_data = unsup_data

        return [_sup_data, _unsup_data]

    def _sequential_data(data_group, n_sup_samples, n_unsup_samples):
        sup_data, unsup_data = data_group

        if 0 < n_sup_samples < len(sup_data):
            _sup_data = sup_data[:n_sup_samples]
        else:
            _sup_data = sup_data

        if 0 < n_unsup_samples < len(unsup_data):
            _unsup_data = unsup_data[:n_unsup_samples]
        else:
            _unsup_data = unsup_data

        return [_sup_data, _unsup_data]

    if shuffle:
        sentiment_data_group = _shuffle_data(
            sentiment_data_group, n_sup_samples, n_unsup_samples
        )
        paraphrase_data_group = _shuffle_data(
            paraphrase_data_group, n_sup_samples, n_unsup_samples
        )
        similarity_data_group = _shuffle_data(
            similarity_data_group, n_sup_samples, n_unsup_samples
        )
    else:
        sentiment_data_group = _sequential_data(
            sentiment_data_group, n_sup_samples, n_unsup_samples
        )
        paraphrase_data_group = _sequential_data(
            paraphrase_data_group, n_sup_samples, n_unsup_samples
        )
        similarity_data_group = _sequential_data(
            similarity_data_group, n_sup_samples, n_unsup_samples
        )

    logger.info(
        f"Sampled {split} SST dataset has {len(sentiment_data_group[0])} supervised rows, "
        f"{len(sentiment_data_group[1])} unsupervised rows."
    )
    logger.info(
        f"Sampled {split} Para dataset has {len(paraphrase_data_group[0])} supervised rows,"
        f"{len(paraphrase_data_group[1])} unsupervised rows."
    )
    logger.info(
        f"Sampled {split} STS dataset has {len(similarity_data_group[0])} supervised rows,"
        f"{len(similarity_data_group[1])} unsupervised rows."
    )

    sentiment_dataloader = create_custom_data_loader(
        sentiment_data_group,
        SentenceClassificationDataset,
        n_batches,
        n_sup_batch_size,
        n_unsup_batch_size,
        num_workers=num_workers,
    )

    paraphrase_dataloader = create_custom_data_loader(
        paraphrase_data_group,
        SentencePairDataset,
        n_batches,
        n_sup_batch_size,
        n_unsup_batch_size,
        num_workers=num_workers,
    )

    similarity_dataloader = create_custom_data_loader(
        similarity_data_group,
        SentencePairDatasetReg2Cls,
        n_batches,
        n_sup_batch_size,
        n_unsup_batch_size,
        num_workers=num_workers,
    )

    return {
        "sst": sentiment_dataloader,
        "para": paraphrase_dataloader,
        "sts": similarity_dataloader,
    }


def create_custom_data_loader(
    data_group, dataset_class, n_batches, n_sup_batch_size, n_unsup_batch_size, **kwargs
):
    combined_dataset = CombinedDataset(*data_group, dataset_class=dataset_class)

    # Initialize custom batch sampler
    batch_sampler = CustomBatchSampler(
        sup_ds_len=len(data_group[0]),
        unsup_ds_len=len(data_group[1]),
        n_batches=n_batches,
        n_sup_batch_size=n_sup_batch_size,
        n_unsup_batch_size=n_unsup_batch_size,
    )

    # Initialize DataLoader with custom batch sampler and collate function
    data_loader = DataLoader(
        combined_dataset,
        batch_sampler=batch_sampler,
        collate_fn=combined_dataset.collate_fn,
        **kwargs,
    )
    return data_loader


class CombinedDataset(Dataset):
    def __init__(self, dataset1, dataset2, dataset_class):
        """dataset 2 is unsupervised data, with corresponding pairs"""
        dataset2_0, dataset2_1 = zip(*dataset2)
        union_dataset = dataset1 + list(dataset2_0) + list(dataset2_1)
        self.len_ds1 = len(dataset1)
        self.len_ds2 = len(dataset2_0)

        self.ds2_start_idx = self.len_ds1
        self.ds3_start_idx = self.len_ds1 + self.len_ds2

        self.dataset = dataset_class(union_dataset)

    def __len__(self):
        return min(self.len_ds1, self.len_ds2)

    def __getitem__(self, idx):
        """
        Assuming idx is a tuple (ds1_idx, ds2_idx, ds3_idx)
        """
        ds1_idx, ds2_idx, ds3_idx = idx
        if ds1_idx is not None:
            return self.dataset[ds1_idx]
        elif ds2_idx is not None:
            return self.dataset[self.ds2_start_idx + ds2_idx]
        elif ds3_idx is not None:
            try:
                return self.dataset[self.ds3_start_idx + ds3_idx]
            except:
                return
        else:
            raise ValueError(f"idx {idx} is invalid")

    def collate_fn(self, all_data):
        return self.dataset.collate_fn(all_data)


class CustomBatchSampler:
    def __init__(
        self,
        sup_ds_len,
        unsup_ds_len,
        n_batches,
        n_sup_batch_size,
        n_unsup_batch_size,
    ):
        self.sup_ds_len = sup_ds_len
        self.unsup_ds_len = unsup_ds_len
        self.n_batches = n_batches

        self.sup_batch_size = n_sup_batch_size
        self.unsup_batch_size = n_unsup_batch_size

        # sample with replacement
        _sup_iter = cycle(range(self.sup_ds_len))
        self.sup_indices = [
            next(_sup_iter) for _ in range(n_batches * n_sup_batch_size)
        ]
        _unsup_iter = cycle(range(self.unsup_ds_len))
        self.unsup_indices = [
            next(_unsup_iter) for _ in range(n_batches * n_unsup_batch_size)
        ]

    def __iter__(self):
        # first sample supervised dataset, then unsupervised and its augmentation
        for i in range(self.n_batches):
            sup_start = i * self.sup_batch_size
            sup_end = sup_start + self.sup_batch_size
            unsup_start = i * self.unsup_batch_size
            unsup_end = unsup_start + self.unsup_batch_size

            sup_batch = self.sup_indices[sup_start:sup_end]
            unsup_batch = self.unsup_indices[unsup_start:unsup_end]

            batch = []
            batch += [(_idx, None, None) for _idx in sup_batch]
            batch += [(None, _idx, None) for _idx in unsup_batch]
            batch += [(None, None, _idx) for _idx in unsup_batch]

            yield batch

    def __len__(self):
        return self.n_batches
