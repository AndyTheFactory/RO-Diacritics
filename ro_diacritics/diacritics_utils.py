"""Utility constants and helper functions for Romanian diacritics processing."""

import re

LOG_NAME = "ro-diacritics"

MAP_DIACRITICS = {
    "ă": "a",
    "â": "a",
    "Â": "A",
    "Ă": "A",
    "ț": "t",
    "Ț": "T",
    "ș": "s",
    "Ș": "S",
    "î": "i",
    "Î": "I",
    "ş": "s",
    "Ş": "S",
    "ţ": "t",
    "Ţ": "T",
}
ALL_CANDIDATES = {
    "ă",
    "â",
    "a",
    "Ă",
    "Â",
    "A",
    "ț",
    "Ț",
    "ș",
    "Ș",
    "î",
    "Î",
    "i",
    "t",
    "s",
    "ş",
    "Ş",
    "ţ",
    "Ţ",
    "S",
    "T",
    "I",
}
DIACRITICS_CANDIDATES = {"a", "i", "s", "t"}
MAP_CORRECT_DIACRITICS = {
    "ş": "ș",
    "Ş": "Ș",
    "ţ": "ț",
    "Ţ": "Ț",
}
MAP_POSSIBLE_CHARS = {
    "a": ["ă", "â", "a"],
    "i": ["î", "i"],
    "s": ["ș", "s"],
    "t": ["ț", "t"],
}

# Pre-compiled regex patterns for efficiency
_RE_CORRECT_DIACRITICS = re.compile(
    "|".join(re.escape(k) for k in MAP_CORRECT_DIACRITICS)
)
_RE_REMOVE_DIACRITICS = re.compile(
    "|".join(re.escape(k) for k in MAP_DIACRITICS)
)
_RE_INTERESTING_CHARS = re.compile(
    "|".join(re.escape(c) for c in DIACRITICS_CANDIDATES)
)


def correct_diacritics(word):
    """Replace old-style cedilla diacritics (ş, Ş, ţ, Ţ) with the correct
    comma-below forms (ș, Ș, ț, Ț) used in modern Romanian orthography.

    :param word: Input string that may contain old-style diacritics.
    :return: String with old-style diacritics replaced by correct forms.
    """
    return _RE_CORRECT_DIACRITICS.sub(
        lambda m: MAP_CORRECT_DIACRITICS[m.group(0)], str(word)
    )


def remove_diacritics(word):
    """Strip all Romanian diacritical characters from *word*, returning the
    plain ASCII base form (e.g. "ș" → "s", "ă" → "a").

    :param word: Input string that may contain Romanian diacritics.
    :return: String with all diacritics removed.
    """
    return _RE_REMOVE_DIACRITICS.sub(
        lambda m: MAP_DIACRITICS[m.group(0)], str(word)
    )


def has_interesting_chars(word):
    """Return True if *word* contains at least one character that could carry
    a Romanian diacritic (i.e. one of 'a', 'i', 's', 't').

    :param word: Input string to check.
    :return: True if a diacritic-candidate character is found, False otherwise.
    """
    return _RE_INTERESTING_CHARS.search(str(word)) is not None


__all__ = [
    "correct_diacritics",
    "remove_diacritics",
    "has_interesting_chars",
    "MAP_DIACRITICS",
    "ALL_CANDIDATES",
    "DIACRITICS_CANDIDATES",
    "MAP_CORRECT_DIACRITICS",
    "MAP_POSSIBLE_CHARS",
    "LOG_NAME",
]