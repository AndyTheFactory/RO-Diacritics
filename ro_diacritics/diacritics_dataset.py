"""Dataset and vocabulary classes for Romanian diacritics restoration."""

import logging
import os
import pickle as pkl
from collections import Counter
from pathlib import Path

import fasttext
import fasttext.util
import numpy as np
import torch
from nltk.tokenize import TreebankWordTokenizer, PunktSentenceTokenizer
from nltk.tokenize import sent_tokenize, word_tokenize
from torch.utils.data import IterableDataset

from .diacritics_utils import (
    correct_diacritics,
    remove_diacritics,
    has_interesting_chars,
    DIACRITICS_CANDIDATES,
    LOG_NAME,
)

logger = logging.getLogger(LOG_NAME)

FASTTEXT_CACHE_DIR = ".model"
FASTTEXT_MODEL_FILE = "cc.ro.300.bin"


def _load_fasttext_model():
    """Download (if necessary) and load the Romanian fastText binary model.

    The model file is cached in *FASTTEXT_CACHE_DIR*.  ``fasttext.util``
    always downloads to the current working directory, so this function
    temporarily changes the directory and restores it afterwards.

    :return: Loaded :class:`fasttext.FastText._FastText` model object.
    """
    cache_dir = Path(FASTTEXT_CACHE_DIR)
    cache_dir.mkdir(exist_ok=True)
    model_path = cache_dir / FASTTEXT_MODEL_FILE

    if not model_path.exists():
        prev_dir = os.getcwd()
        try:
            os.chdir(cache_dir)
            fasttext.util.download_model("ro", if_exists="ignore")
        finally:
            os.chdir(prev_dir)

    return fasttext.load_model(str(model_path))


class DiacriticsVocab:
    """Word-level vocabulary with pre-computed fastText word embeddings.

    For each token in *distinct_tokens* the corresponding 300-dimensional
    fastText word vector is stored.  Because fastText supports sub-word
    information, out-of-vocabulary words still receive a meaningful vector.

    :param distinct_tokens: :class:`~collections.Counter` of token frequencies
        collected from the training corpus.
    :param max_vocab: Maximum number of vocabulary tokens (most frequent ones
        are kept).
    :param pad_token: Integer index used for the padding pseudo-token.
    :param unk_token: Integer index used for the unknown-word pseudo-token.
    :param max_char_vocab: Maximum Unicode code-point treated by the character
        vocabulary; code-points above this are mapped to *overflow_char*.
    :param overflow_char: Integer encoding used for out-of-range characters.
    """

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

        ft = _load_fasttext_model()

        distinct_tokens = dict(distinct_tokens.most_common(max_vocab))

        self.vocab["itos"] = dict(enumerate(distinct_tokens, 1))
        self.vocab["itos"][self.pad_token] = "<pad>"
        self.vocab["itos"][self.unk_token] = "<unk>"

        self.vocab["stoi"] = {v: k for k, v in self.vocab["itos"].items()}

        for word, idx in self.vocab["stoi"].items():
            if word in ("<pad>", "<unk>"):
                continue
            vec = ft.get_word_vector(word)
            self.vocab["vectors"][idx] = torch.tensor(vec, dtype=torch.float32)

    def encode_char(self, c):
        """Return the integer encoding for character *c*.

        Characters whose Unicode code-point exceeds *max_char_vocab* are
        mapped to *overflow_char*.

        :param c: Single character to encode.
        :return: Integer code-point or *overflow_char*.
        """
        return ord(c) if ord(c) <= self.max_char_vocab else self.overflow_char


class DiacriticsDataset(IterableDataset):
    """Iterable PyTorch dataset for Romanian diacritics restoration.

    Reads tokenised Romanian text and yields *(inputs, label)* tuples where
    *inputs* is a triple of ``(char_tensor, word_embedding, sentence_embedding)``
    and *label* is a one-hot tensor of shape ``(3,)`` representing the correct
    diacritic class for a candidate character position.

    :param data: Path to a plain-text file, a ``.pkl`` cache file produced by
        a previous run, or a raw text string.  Pass an empty string when only
        running inference with a pre-built *diacritics_vocab*.
    :param character_window: Half-width of the character context window (the
        total window is ``2 * character_window + 1`` characters).
    :param sentence_window: Width of the sliding sentence window in words.
    :param min_line_length: Lines (or sentences) shorter than this are
        discarded during loading.
    :param max_vocab: Maximum vocabulary size (most frequent tokens kept).
    :param max_char_vocab: Maximum Unicode code-point handled by the character
        vocabulary; higher code-points are mapped to *overflow_char*.
    :param overflow_char: Fallback integer for out-of-range characters.
    :param diacritics_vocab: Pre-built :class:`DiacriticsVocab` to reuse
        (e.g. loaded from a saved checkpoint).  When ``None`` a new vocabulary
        is built from *data* using fastText embeddings.
    """

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
        """Return the underlying vocabulary dictionary (``itos``, ``stoi``, ``vectors``)."""
        return self.diacritics_vocab.vocab

    def encode_char(self, c):
        """Encode a single character using the vocabulary's character mapping.

        :param c: Single character to encode.
        :return: Integer code-point, capped at *max_char_vocab*.
        """
        return self.diacritics_vocab.encode_char(c)

    def __iter__(self):
        """Yield training samples by iterating over all stored sentences."""
        yield from self.parse_text()

    @staticmethod
    def get_label(original_char):
        """Convert a (possibly diacritical) character to a one-hot class label.

        :param original_char: Lowercase character from the original text.
        :return: One-hot :class:`torch.Tensor` of shape ``(3,)`` where index 0
            means "no diacritic", 1 means "first candidate diacritic" (ă, î,
            ș, ț), and 2 means "second candidate diacritic" (â).
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
        """Map a base character and a predicted class label back to the
        (possibly diacritical) output character.

        :param original_char: Lowercase base character (e.g. ``'a'``).
        :param label: Predicted integer class (0 = no change, 1 = first
            candidate, 2 = second candidate).
        :return: The correctly diacritised character, or *original_char* if no
            transformation applies.
        """
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
        """Build character-level input encodings for every diacritic-candidate
        position in the token at *token_idx*.

        For each candidate character a fixed-width context window of encoded
        characters is produced: *character_window* characters to the left of
        the candidate (padded with zeros), the candidate itself, and
        *character_window* characters to the right (padded with zeros).

        :param line: List of word strings (diacritics stripped, lower-cased).
        :param line_orig: List of word strings in their original form (used to
            determine the ground-truth label).
        :param token_idx: Index of the target token inside *line*.
        :return: Tuple of three lists aligned by position:

            * ``encoded_list`` – list of integer lists, each of length
              ``2 * character_window + 1``.
            * ``labels`` – list of one-hot label tensors.
            * ``char_positions`` – list of intra-word character offsets.
        """
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
        """Look up the pre-computed word embedding for *word*.

        Falls back to the ``<unk>`` vector when *word* is not in the
        vocabulary.

        :param word: Token string to look up.
        :return: Embedding :class:`torch.Tensor` of shape ``(word_embedding_size,)``.
        """
        if word in self.vocab["stoi"]:
            idx = self.vocab["stoi"][word]
            embed = self.vocab["vectors"][idx]
        else:
            embed = self.vocab["vectors"][self.unk_token]

        return embed

    def get_sentence_emb(self, sentence):
        """Build a sentence-level embedding matrix for a window of words.

        Short sentences are right-padded with the ``<pad>`` vector so the
        result always has *sentence_window* rows.

        :param sentence: List of token strings.
        :return: :class:`torch.Tensor` of shape
            ``(sentence_window, word_embedding_size)``.
        """
        encoded = list(map(self.get_word_emb, sentence))
        if len(encoded) < self.sentence_window:
            encoded = encoded + [self.get_word_emb("<pad>")] * (
                self.sentence_window - len(encoded)
            )
        return torch.stack(encoded)

    def parse_text(self):
        """Generate *(inputs, label)* training samples from :attr:`texts`.

        For every sentence in the corpus and every diacritic-candidate
        character, a triple of ``(char_tensor, word_embedding,
        sentence_embedding)`` is yielded together with the corresponding
        one-hot label tensor.

        :yields: Tuple of ``((char_tensor, word_emb, sentence_emb), label)``.
        """
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
        """Build inference tensors for all diacritic-candidate characters in
        *text*.

        Unlike :meth:`parse_text`, this method does not require ground-truth
        labels and returns the tensors together with the absolute character
        positions in *text* so predictions can later be mapped back.

        :param text: Raw input text to restore diacritics for.
        :param stride: Step size when sliding the sentence window over the
            token sequence.  Values greater than 1 reduce computation at the
            cost of receiving fewer predictions per character (which are
            averaged during post-processing).
        :return: Tuple of two aligned lists:

            * ``input_tensors`` – list of ``[char_tensor, word_emb,
              sentence_emb]`` triples.
            * ``character_indices`` – list of absolute character offsets into
              *text* corresponding to each entry in *input_tensors*.
        """
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
        """Load and pre-process text data from a file or raw string.

        Supports three input formats:

        * ``.pkl`` file – a previously pickled ``(texts, counter)`` tuple is
          loaded directly.
        * Plain-text file – lines are sentence-tokenised and then
          word-tokenised; lines shorter than *min_line_length* are dropped.
        * Raw text string – same sentence/word tokenisation is applied inline.

        :param data: File path (str or :class:`~pathlib.Path`) or raw text.
        :param min_line_length: Minimum character length for a sentence to be
            kept.
        :return: Tuple of ``(texts, distinct_no_diacritics)`` where *texts* is
            a list of token lists and *distinct_no_diacritics* is a
            :class:`~collections.Counter` of diacritics-free token frequencies.
        """
        if data and Path(data).exists():
            filename = data
            if Path(filename).suffix.lower() == ".pkl":  # loading cached pickle
                with open(filename, "rb") as f:
                    texts, distinct_no_diacritics = pkl.load(f)
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
