"""Text normalization for the full-text index: folding, stopwords and light suffix stemming.

German ("de") is the original rule set and is kept unchanged. English ("en") is a light, conservative
suffix stripper (a small subset of Porter step 1). With several languages active the rule is:

* stopwords: the union of the stopword lists of all active languages;
* stemming: the stemmers run in a fixed order (German first, then English). The first stemmer that
  changes the word wins; the remaining stemmers do not run.

Index time and query time must use the same rules. The core stores `NORM_VERSION` and the active
language list in the index metadata and rebuilds the full-text table when either changes.
"""
import re
from functools import lru_cache

# Bump when a rule below changes in a way that changes produced terms.
NORM_VERSION = "1"
SUPPORTED = ("de", "en")
STEM_ORDER = ("de", "en")

# ---------------------------------------------------------------- German (unchanged from the original engine)
STOP_DE = set("""der die das den dem des ein eine einen einem einer eines und oder aber auch nicht kein keine
ist sind war waren wird werden wurde wurden hat haben hatte hatten sein ihr ihre ihm ihn er sie es wir ich du
von zu zum zur mit bei nach aus auf an in im am um fuer ueber unter vor durch gegen ohne bis wie was wer wo wann
warum wieso welche welcher welches welchen wenn dass als so noch nur schon mal man kann koennen muss muessen
soll sollen gibt gibts dann dort hier dies diese dieser dieses diesen the of to and for is in on at by with""".split())

SUFFIXES_DE = ("ungen", "ung", "heiten", "heit", "keiten", "keit", "schaft", "ern", "ers", "en", "er", "es", "em", "e", "s", "n")


def stem_de(w):
    if len(w) <= 4 or w.isdigit():
        return w
    for suf in SUFFIXES_DE:
        if w.endswith(suf) and len(w) - len(suf) >= 4:
            return w[:-len(suf)]
    return w


# ---------------------------------------------------------------- English (light, conservative)
STOP_EN = set("""a an the and or but if then else of to in on at by for with from into onto about as is are was were
be been being am do does did doing have has had having it its this that these those there here what which who whom
whose when where why how not no nor so than too very can could will would shall should may might must i me my we
our you your he him his she her they them their all any each both some such only own same just also""".split())

_VOWELS = set("aeiouy")


def _has_vowel(s):
    return any(c in _VOWELS for c in s)


def stem_en(w):
    """Plural and -ed/-ing stripping plus a final y -> i (as in Porter step 1c), so that
    query/queries and index/indexes/indexed/indexing share a stem. Stems keep at least 3 letters."""
    if len(w) <= 3 or w.isdigit():
        return w
    out = w
    if out.endswith("sses"):
        out = out[:-2]
    elif out.endswith("ies") and len(out) > 4:
        out = out[:-3] + "i"
    elif out.endswith(("ches", "shes", "xes", "zes")) and len(out) - 2 >= 3:
        out = out[:-2]
    elif out.endswith(("ss", "us", "is")):
        pass
    elif out.endswith("s") and len(out) - 1 >= 3:
        out = out[:-1]
    else:
        for suf in ("ing", "ed"):
            base = out[:-len(suf)]
            if out.endswith(suf) and len(base) >= 3 and _has_vowel(base):
                if len(base) >= 4 and base[-1] == base[-2] and base[-1] not in "aeiouylsz":
                    base = base[:-1]          # running -> run, stopped -> stop
                out = base
                break
    if len(out) >= 4 and out.endswith("y") and out[-2] not in _VOWELS:
        out = out[:-1] + "i"
    return out


STOPWORDS = {"de": STOP_DE, "en": STOP_EN}
STEMMERS = {"de": stem_de, "en": stem_en}

# ---------------------------------------------------------------- shared
_FOLD = str.maketrans({"ä": "a", "ö": "o", "ü": "u", "é": "e", "è": "e", "á": "a", "à": "a"})


def fold(s):
    s = s.casefold().replace("ß", "ss")
    return s.translate(_FOLD)


TOKEN = re.compile(r"[a-z0-9]+")


def check_languages(languages):
    langs = [l.strip().lower() for l in languages if l and l.strip()]
    bad = [l for l in langs if l not in SUPPORTED]
    if bad:
        raise ValueError(f"unsupported language(s) {bad}; supported: {list(SUPPORTED)}")
    if not langs:
        raise ValueError("at least one language must be active")
    return tuple(l for l in STEM_ORDER if l in langs)


class Normalizer:
    """Turns text into index terms for one fixed set of languages."""

    def __init__(self, languages=("de", "en")):
        self.languages = check_languages(languages)
        self.stop = set().union(*(STOPWORDS[l] for l in self.languages))
        stemmers = [STEMMERS[l] for l in self.languages]

        @lru_cache(maxsize=200_000)   # pure function; snippet scoring calls it very often
        def stem(w):
            for fn in stemmers:
                s = fn(w)
                if s != w:
                    return s
            return w

        self.stem = stem

    @property
    def signature(self):
        """Stored in the index; a change forces a rebuild of the full-text table."""
        return f"{NORM_VERSION}:{','.join(self.languages)}"

    def terms(self, text, drop_stop=True):
        out = []
        for t in TOKEN.findall(fold(text)):
            if drop_stop and (t in self.stop or len(t) < 2):
                continue
            out.append(self.stem(t))
        return out
