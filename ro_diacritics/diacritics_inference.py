import zipfile
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset
from collections import OrderedDict
from urllib.request import urlretrieve

from tqdm import tqdm

from .diacritics_dataset import DiacriticsDataset
from .diacritics_model import Diacritics
from .diacritcs_train import predict
from . import diacritics_dataset
from .diacritics_utils import LOG_NAME

logger = logging.getLogger(LOG_NAME)

_model: Diacritics = None
_dataset: DiacriticsDataset = None

CACHED_DIR = ".model"
MODEL_FILE = "diacritice.pt"
MODEL_URL = (
    "https://github.com/AndyTheFactory/andythefactory.github.io/raw/main/diacritice.zip"
)


def get_cached_model():
    """
    Loads the cached model. If not, it tries to download the model from github
    :return: local path to model
    """
    filename = Path(CACHED_DIR) / MODEL_FILE
    if filename.exists():
        return filename
    else:

        def reporthook(t):
            last_b = [0]

            def inner(b=1, bsize=1, tsize=None):
                if tsize is not None:
                    t.total = tsize
                t.update((b - last_b[0]) * bsize)
                last_b[0] = b

            return inner

        Path(CACHED_DIR).mkdir(exist_ok=True)
        dest_file = Path(CACHED_DIR) / Path(MODEL_URL).name
        with tqdm(unit="B", unit_scale=True, miniters=1, desc=str(filename)) as t:
            try:
                urlretrieve(MODEL_URL, dest_file, reporthook=reporthook(t))
            except KeyboardInterrupt as e:  # remove the partial zip file
                dest_file.unlink(missing_ok=True)
                raise e
        if dest_file.suffix == ".zip":
            with zipfile.ZipFile(dest_file, "r") as zf:
                zf.extractall(CACHED_DIR)
        return filename


def load_model(filename) -> (Diacritics, DiacriticsDataset):
    """
    Loads a trained :class:Diacritics model from cached file.
    Also, used hyperparams are loaded from the file
    :param filename: local path to stored model
    :return: loaded :class:Diacritics object and the used vocabulary (must be the same as in training)
    """
    import sys

    sys.modules["diacritics_dataset"] = diacritics_dataset

    checkpoint = torch.load(filename, map_location="cpu")
    params = checkpoint["hyperparams"]
    model = Diacritics(
        nr_classes=params["nr_classes"],
        word_embedding_size=params["word_embedding_size"],
        character_embedding_size=params["character_embedding_size"],
        char_vocabulary_size=params["char_vocabulary_size"],
        char_padding_index=params["char_padding_index"],
        character_window=params["character_window"],
        sentence_window=params["sentence_window"],
        characters_lstm_size=params["characters_lstm_size"],
        sentence_lstm_size=params["sentence_lstm_size"],
    )
    model.load_state_dict(checkpoint["model_state"])

    checkpoint["valid_f1"] = checkpoint["valid_f1"] if "valid_f1" in checkpoint else 0
    logger.info(
        f"Loaded checkpoint: Epoch: {checkpoint['epoch']}, valid_acc: {checkpoint['valid_acc']}, valid_f1: {checkpoint['valid_f1']}"
    )

    return model, checkpoint["vocabulary"]


def initmodel(filename=None):
    global _model, _dataset
    if filename is None:
        filename = get_cached_model()
    _model, vocab = load_model(filename)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        logger.warning("GPU not available, using CPU")
    else:
        logger.info("GPU available")
    _model = _model.to(device)
    _dataset = DiacriticsDataset(
        "", _model.character_window, _model.sentence_window, diacritics_vocab=vocab
    )


def restore_diacritics(text, batch_size=128):
    """
    Transforms a Romanian text without diacritics (“ă”, “â”, “î”, “ș”, and “ț”) into a
    proper spelled text with romanian character set. The casing of the text is preserved.

    :param text: input text (without diacritics)
    :param batch_size: Adjust the batch size according to the available memory.
    :return: text with diacrtics replaced
    """
    if _model is None:
        initmodel()

    input_tensors, character_indices = _dataset.gen_batch(text, stride=10)

    class DS(IterableDataset):
        def __init__(self, input_tensors):
            self.input_tensors = input_tensors

        def __iter__(self):
            return iter(self.input_tensors)

    input_data = DataLoader(DS(input_tensors), batch_size=batch_size)

    predictions = predict(_model, input_data)

    prediction_tuples = sorted(zip(character_indices, predictions), key=lambda x: x[0])

    d = OrderedDict()

    for k, v in prediction_tuples:
        d.setdefault(k, []).append(v)

    text = list(text)
    for idx, p in d.items():
        p = np.argmax(np.average(p, axis=0))
        chr = DiacriticsDataset.get_char_from_label(text[idx].lower(), p)
        if text[idx].lower() == text[idx]:
            if chr == "â" and (
                idx == 0
                or idx == len(text) - 1
                or not text[idx - 1].isalpha()
                or not text[idx + 1].isalpha()
            ):
                #  â is not admitted at the beginning or end
                continue
            if chr == "î" and (
                idx > 0
                and idx < len(text) - 1
                and text[idx - 1].isalpha()
                and text[idx + 1].isalpha()
            ):
                # î is not admitted inside a word
                continue

            text[idx] = chr
        else:
            text[idx] = chr.upper()

    return "".join(text)
