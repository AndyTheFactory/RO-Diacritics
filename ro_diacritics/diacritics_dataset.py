import pickle as pkl
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from nltk.tokenize import TreebankWordTokenizer, PunktSentenceTokenizer
from nltk.tokenize import sent_tokenize, word_tokenize
from torch.utils.data import IterableDataset
from torchtext.vocab import FastText

from .diacritics_utils import (
    correct_diacritics,
    remove_diacritics,
    has_interesting_chars,
    DIACRITICS_CANDIDATES,
)


class DiacriticsVocab:
    def __init__(
        self,
        distinct_tokens: Counter,
        max_vocab,
        pad_token,
        unk_token,
        max_char_vocab,
        overflow_char,
    ):
        self.vocab = {
            "itos": {},
            "stoi": {},
            "vectors": torch.zeros(max_vocab + 2, 300, dtype=torch.float32),
        }
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.max_char_vocab = max_char_vocab
        self.overflow_char = overflow_char

        embedding = FastText("ro")
        distinct_tokens = dict(distinct_tokens.most_common(max_vocab))

        self.vocab["itos"] = dict(enumerate(distinct_tokens, 1))
        self.vocab["itos"][self.pad_token] = "<pad>"
        self.vocab["itos"][self.unk_token] = "<unk>"

        self.vocab["stoi"] = {v: k for k, v in self.vocab["itos"].items()}

        fasttext_counts = Counter([remove_diacritics(word) for word in embedding.stoi])

        for (
            word,
            index,
        ) in embedding.stoi.items():  # Aggregate embeddings with diacritics
            word = remove_diacritics(word)
            if word in self.vocab["stoi"]:
                idx = self.vocab["stoi"][word]
                self.vocab["vectors"][idx] = (
                    self.vocab["vectors"][idx] + embedding.vectors[index]
                )

        for word, index in self.vocab["stoi"].items():
            if fasttext_counts[word] > 1:
                self.vocab["vectors"][index] = (
                    self.vocab["vectors"][index] / fasttext_counts[word]
                )

    def encode_char(self, c):
        return ord(c) if ord(c) <= self.max_char_vocab else self.overflow_char


class DiacriticsDataset(IterableDataset):
    def __init__(
        self,
        data,
        character_window,
        sentence_window,
        min_line_length=50,
        max_vocab=25000,
        max_char_vocab=770,
        overflow_char=255,
        diacritics_vocab: DiacriticsVocab = None,
    ):
        """

        :param data: Textfile, Pickle file or raw text
        :param character_window:
        :param sentence_window:
        :param min_line_length:
        :param max_vocab:
        :param max_char_vocab:
        :param overflow_char:
        :param diacritics_vocab:
        """
        self.character_window = character_window
        self.sentence_window = sentence_window
        self.min_line_length = min_line_length
        self.texts, distinct_tokens = self.load_texts(data, self.min_line_length)
        self.max_vocab = max_vocab
        self.pad_character = 0

        if diacritics_vocab is None:
            self.max_char_vocab = max_char_vocab
            self.overflow_char = overflow_char
            self.pad_token = 0
            self.unk_token = max_vocab + 1
        else:
            self.max_char_vocab = diacritics_vocab.max_char_vocab
            self.overflow_char = diacritics_vocab.overflow_char
            self.pad_token = diacritics_vocab.pad_token
            self.unk_token = diacritics_vocab.unk_token

        self.diacritics_vocab = (
            DiacriticsVocab(
                distinct_tokens,
                self.max_vocab,
                self.pad_token,
                self.unk_token,
                self.max_char_vocab,
                self.overflow_char,
            )
            if diacritics_vocab is None
            else diacritics_vocab
        )

    @property
    def vocab(self):
        return self.diacritics_vocab.vocab

    def encode_char(self, c):
        return self.diacritics_vocab.encode_char(c)

    def __iter__(self):
        return self.parse_text()

    @staticmethod
    def get_label(original_char):
        """
        :param original_char: lowercase diacritics char
        :return: 0 if no change (not turing into diacritics),
                 1 if changed to first candidate of diacritics
                 2 if changed to second candidate of diacritics (a has 2 candidates)
        """
        diacritic_to_label = {
            "ă": 1,
            "â": 2,
            "î": 1,
            "ș": 1,
            "ț": 1,
        }
        label = (
            diacritic_to_label[original_char]
            if original_char in diacritic_to_label
            else 0
        )
        label_tensor = torch.eye(3)

        return label_tensor[label]

    @staticmethod
    def get_char_from_label(original_char, label):
        if original_char not in DIACRITICS_CANDIDATES:
            return original_char

        if label == 1:
            return {
                "a": "ă",
                "i": "î",
                "s": "ș",
                "t": "ț",
            }[original_char]
        elif label == 2:
            return "â"
        else:
            return original_char

    def get_char_input(self, line, line_orig, token_idx):
        prefix_s = " ".join(line[:token_idx])
        suffix_s = " ".join(line[token_idx + 1 :])
        word = line[token_idx]
        word_orig = line_orig[token_idx].lower()
        encoded_list = []
        labels = []
        char_positions = []

        for ix, c in enumerate(word):
            if c in DIACRITICS_CANDIDATES:
                encode_text = (prefix_s + " " + word[:ix]).strip()
                encode_text = encode_text[-self.character_window :]
                l = (
                    [self.pad_character] * (self.character_window - len(encode_text))
                    + list(map(self.encode_char, encode_text))
                    + [self.encode_char(c)]
                )
                encode_text = (word[ix + 1 :] + " " + suffix_s).strip()
                encode_text = encode_text[: self.character_window]
                l = (
                    l
                    + list(map(self.encode_char, encode_text))
                    + [self.pad_character] * (self.character_window - len(encode_text))
                )
                encoded_list.append(l)
                labels.append(self.get_label(word_orig[ix]))
                char_positions.append(ix)
        return encoded_list, labels, char_positions

    def get_word_emb(self, word):
        if word in self.vocab["stoi"]:
            idx = self.vocab["stoi"][word]
            embed = self.vocab["vectors"][idx]
        else:
            embed = self.vocab["vectors"][self.unk_token]

        return embed

    def get_sentence_emb(self, sentence):
        encoded = list(map(self.get_word_emb, sentence))
        if len(encoded) < self.sentence_window:
            encoded = encoded + [self.get_word_emb("<pad>")] * (
                self.sentence_window - len(encoded)
            )
        return torch.stack(encoded)

    def parse_text(self):
        "Generates one sample of data"
        # Select sample
        for line_orig in self.texts:
            line = [
                remove_diacritics(x).lower() for x in line_orig
            ]  # embeddings based on words without diacritics
            line_indices = np.arange(len(line))
            if len(line_indices) > self.sentence_window:
                windows = np.lib.stride_tricks.sliding_window_view(
                    line_indices, self.sentence_window
                )
            else:
                windows = [line_indices]
            for window in windows:
                sentence_emb = self.get_sentence_emb(line[window[0] : window[-1] + 1])
                for token_idx in window:
                    if has_interesting_chars(line[token_idx]):
                        char_inputs, labels, _ = self.get_char_input(
                            line, line_orig, token_idx
                        )
                        word_emb = self.get_word_emb(line[token_idx])
                        for ix, char_input in enumerate(char_inputs):
                            yield (
                                torch.tensor(char_input),
                                word_emb,
                                sentence_emb,
                            ), labels[ix]

    def gen_batch(self, text, stride=1):
        """

        :param text: input text to be processed
        :param stride: generate sentence windows with this stride (you have to pool the results since you will
                        get more than 1 prediciton per character)
        :return:
        """
        text_plain = remove_diacritics(text).lower()
        lines = PunktSentenceTokenizer().span_tokenize(text)
        character_indices = []
        input_tensors = []

        for line_span in lines:
            line = text[line_span[0] : line_span[1]]
            words = list(TreebankWordTokenizer().span_tokenize(line))
            word_indices = np.arange(len(words))
            line_tokens = [line[x[0] : x[1]] for x in words]
            if len(words) > self.sentence_window:
                windows = np.lib.stride_tricks.sliding_window_view(
                    word_indices, self.sentence_window
                )[::stride, :]
                # add a reminder window (if it does not fit perfectly)
                if windows.max() < max(word_indices):
                    windows = np.vstack(
                        [windows, word_indices[-self.sentence_window :]]
                    )
            else:
                windows = [word_indices]

            for window in windows:
                sentence = [
                    line[x[0] : x[1]] for x in words[window[0] : window[-1] + 1]
                ]
                sentence_emb = self.get_sentence_emb(sentence)
                for token_idx in window:
                    word = line[words[token_idx][0] : words[token_idx][1]]
                    if has_interesting_chars(word):
                        char_inputs, _, char_positions = self.get_char_input(
                            line_tokens, line_tokens, token_idx
                        )
                        word_emb = self.get_word_emb(word)
                        for ix, char_input in enumerate(char_inputs):
                            input_tensors.append(
                                [torch.tensor(char_input), word_emb, sentence_emb]
                            )
                            character_indices.append(
                                line_span[0] + words[token_idx][0] + char_positions[ix]
                            )

        return input_tensors, character_indices

    @staticmethod
    def load_texts(data, min_line_length):
        if data and Path(data).exists():
            filename = data
            if Path(filename).suffix.lower() == ".pkl":  # loading cached pickle
                texts, distinct_no_diacritics = pkl.load(open(filename, "rb"))
            else:
                with open(filename, "r", encoding="utf-8") as f:
                    texts = [
                        correct_diacritics(line)
                        for line in f
                        if len(line) > min_line_length
                    ]

                texts = [
                    x
                    for line in texts
                    for x in sent_tokenize(line, "english")
                    if len(x) > min_line_length
                ]
                texts = [word_tokenize(line) for line in texts]

                distinct_no_diacritics = Counter(
                    [remove_diacritics(x) for line in texts for x in line]
                )
        else:
            texts = [
                x for x in sent_tokenize(data, "english") if len(x) > min_line_length
            ]
            texts = [word_tokenize(line) for line in texts]

            distinct_no_diacritics = Counter(
                [remove_diacritics(x) for line in texts for x in line]
            )

        return texts, distinct_no_diacritics
