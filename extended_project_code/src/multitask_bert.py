"""
Multitask BERT class, starter training code, evaluation, and test code.

Of note are:
* class MultitaskBERT: Your implementation of multitask BERT.
* function train_multitask: Training procedure for MultitaskBERT. Starter code
    copies training procedure from `classifier.py` (single-task SST).
* function test_multitask: Test procedure for MultitaskBERT. This function generates
    the required files for submission.

Running `python multitask_classifier.py` trains and tests your MultitaskBERT and
writes all required submission files.
"""

import torch
from torch import nn
import torch.nn.functional as F
from .bert import BertModel


class SentimentClassifier(nn.Module):
    """
    Sentiment Analysis (predict_sentiment)
    Given a batch of sentences, outputs logits for classifying sentiment.
    There are 5 sentiment classes:
    (0 - negative, 1- somewhat negative, 2- neutral, 3- somewhat positive, 4- positive)
    Thus, your output should contain 5 logits for each sentence.
    """

    def __init__(self, config):
        super(SentimentClassifier, self).__init__()
        self.n_class = 5
        self.model = nn.Sequential(
            nn.Linear(config.hidden_size, self.n_class),
        )

    def forward(self, embeddings):
        return self.model(embeddings)


class ParaphraseClassifier(nn.Module):
    """
    Paraphrase Detection (2-classes)
    Given a batch of pairs of sentences, outputs a single logit for predicting
    whether they are paraphrases.
    Need to output single logit then goes into Sigmoid
    """

    def __init__(self, config):
        super(ParaphraseClassifier, self).__init__()
        self.n_class = 2
        self.model = nn.Sequential(
            nn.Linear(config.hidden_size, self.n_class),
        )

    def forward(self, embeddings):
        return self.model(embeddings)


class TextualSimilarity(nn.Module):
    """
    Paraphrase Detection (2-classes)
    Given a batch of pairs of sentences, outputs a single logit corresponding
    to how similar they are.
    Note that your output should be unnormalized (a logit).
    """

    def __init__(self, config):
        super(TextualSimilarity, self).__init__()
        self.n_class = 26  # 0.0 to 5.0 with 0.2 increments
        self.model = nn.Sequential(
            nn.Linear(config.hidden_size, self.n_class),
        )

    def forward(self, embeddings):
        logits = self.model(embeddings)
        # scaled = F.sigmoid(logits) * 5
        return logits


class MultitaskBERT(nn.Module):
    """
    This module should use BERT for 3 tasks:

    - Sentiment classification (predict_sentiment)
    - Paraphrase detection (predict_paraphrase)
    - Semantic Textual Similarity (predict_similarity)
    """

    def __init__(self, config, fine_tune_mode):
        super(MultitaskBERT, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        # last-linear-layer mode does not require updating BERT paramters.
        assert fine_tune_mode in ["last-linear-layer", "full-model"]
        for param in self.bert.parameters():
            if fine_tune_mode == "last-linear-layer":
                param.requires_grad = False
            elif fine_tune_mode == "full-model":
                param.requires_grad = True
        # You will want to add layers here to perform the downstream tasks.
        self.sentiment_classifier = SentimentClassifier(config)
        self.paraphrase_classifier = ParaphraseClassifier(config)
        self.similarity_classifier = TextualSimilarity(config)

    def forward(self, input_ids, attention_mask):
        """Takes a batch of sentences and produces embeddings for them."""
        # The final BERT embedding is the hidden state of [CLS] token (the first token)
        # Here, you can start by just returning the embeddings straight from BERT.
        # When thinking of improvements, you can later try modifying this
        # (e.g., by adding other layers).

        # Get the BERT embeddings for the input_ids
        bert_output = self.bert(input_ids, attention_mask)
        # Get the hidden states from the output
        hs_cls = bert_output["pooler_output"]

        return hs_cls

    def predict_sentiment(self, input_ids, attention_mask):
        """Given a batch of sentences, outputs logits for classifying sentiment.
        There are 5 sentiment classes:
        (0 - negative, 1- somewhat negative, 2- neutral,
        3- somewhat positive, 4- positive)
        Thus, your output should contain 5 logits for each sentence.
        """
        embed = self.forward(input_ids, attention_mask)
        return self.sentiment_classifier(embed)

    def predict_paraphrase(
        self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2
    ):
        """Given a batch of pairs of sentences, outputs a single logit for predicting
        whether they are paraphrases.
        Note that your output should be unnormalized (a logit); it will be passed to
        the sigmoid function during evaluation.
        """
        input_ids, attention_mask = self.combine_two_inputs_using_sep(
            input_ids_1, attention_mask_1, input_ids_2, attention_mask_2
        )
        combined_embed = self.forward(input_ids, attention_mask)
        return self.paraphrase_classifier(combined_embed)

    def predict_similarity(
        self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2
    ):
        """Given a batch of pairs of sentences, outputs a single logit corresponding to
        how similar they are.
        Note that your output should be unnormalized (a logit).
        """
        input_ids, attention_mask = self.combine_two_inputs_using_sep(
            input_ids_1, attention_mask_1, input_ids_2, attention_mask_2
        )
        combined_embed = self.forward(input_ids, attention_mask)
        return self.similarity_classifier(combined_embed)

    @staticmethod
    def combine_two_inputs_using_sep(
        input_ids_1, attention_mask_1, input_ids_2, attention_mask_2, max_length=512
    ):
        def _trim_input(input_ids, attention_mask):
            if input_ids.size(-1) > max_length // 2:
                input_ids = input_ids[..., : max_length // 2]
                attention_mask = attention_mask[..., : max_length // 2]
            return input_ids, attention_mask

        # change input_ids_2 CLS (101) to SEP (102)
        input_ids_2[:, 0] = 102
        # Combine the two input_ids and attention_masks

        input_ids_1, attention_mask_1 = _trim_input(input_ids_1, attention_mask_1)
        input_ids_2, attention_mask_2 = _trim_input(input_ids_2, attention_mask_2)

        input_ids = torch.cat([input_ids_1, input_ids_2], dim=1)
        attention_mask = torch.cat([attention_mask_1, attention_mask_2], dim=1)
        return input_ids, attention_mask

    @staticmethod
    def combine_two_embeddings_dot_product(embeddings_1, embeddings_2):
        combined = embeddings_1.unsqueeze(-1) @ embeddings_2.unsqueeze(-1).transpose(
            1, 2
        )
        return combined

    @staticmethod
    def combine_two_embeddings_simple(embeddings_1, embeddings_2):
        combined = torch.cat([embeddings_1, embeddings_2], dim=-1)
        return combined

    @staticmethod
    def combine_two_embeddings_cosine(embeddings_1, embeddings_2):
        combined = torch.cosine_similarity(embeddings_1, embeddings_2, dim=-1)
        combined = combined.unsqueeze(-1)
        return combined

    @staticmethod
    def combine_two_embeddings_sbert_abs_diff(embeddings_1, embeddings_2):
        combined = torch.abs(embeddings_1 - embeddings_2)
        return combined
