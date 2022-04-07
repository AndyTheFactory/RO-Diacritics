import re

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


def correct_diacritics(word):
    # use these three lines to do the replacement
    rep = dict((re.escape(k), v) for k, v in MAP_CORRECT_DIACRITICS.items())
    pattern = re.compile("|".join(rep.keys()))
    return pattern.sub(lambda m: rep[re.escape(m.group(0))], str(word))


def remove_diacritics(word):
    # use these three lines to do the replacement
    rep = dict((re.escape(k), v) for k, v in MAP_DIACRITICS.items())
    pattern = re.compile("|".join(rep.keys()))
    return pattern.sub(lambda m: rep[re.escape(m.group(0))], str(word))


def has_interesting_chars(word):
    # use these three lines to do the replacement
    pattern = re.compile("|".join(DIACRITICS_CANDIDATES))
    return pattern.search(str(word)) is not None
