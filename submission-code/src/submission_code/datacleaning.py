from dataclasses import dataclass
import shlex
from typing import Tuple, List, Dict
import re
replacer_re = re.compile(r"\[-\[(\d*=)?\w+\]-\]")

space_splitter_re = re.compile("( \(|\) |\)$|, | |\\\".*?\\\"|'.*?')")


def normalize_nl(nl: str) -> str:
    _, nl = unquote(nl)
    nl = clean_punkt(nl)
    nl = cheap_replacers(nl)
    nl = word_by_word_expand(nl)
    #nl = replace_files(nl)
    nl = replace_numbers(nl)
    return nl


def replace_cmd(cmd: str) -> str:
    return re.sub(replacer_re, "replval", cmd)


def word_by_word_expand(nl: str):
    out_words = []
    norm_state = NormState()
    for word in pretokenize(nl):
        out_words.extend(multi_expand_word(word, norm_state))
    return " ".join(out_words)


def pretokenize(string) -> List[str]:
    return [p.strip() for p in space_splitter_re.split(string) if p.strip()]


class CountingDict:
    def __init__(self, prefix: str):
        self._conversions = {}
        self._prefix = prefix

    def convert(self, val: str):
        if val in self._conversions:
            return self._conversions[val]
        new_val = f"{self._prefix}{len(self._conversions)}"
        self._conversions[val] = new_val
        return new_val

    def has_seen(self, val: str):
        return val in self._conversions


class NormState:
    def __init__(self):
        self.files_seen = CountingDict("CONSTPATHNAME")
        self.bash_vars = CountingDict("CONSTBASHVAR")


def multi_expand_word(word: str, norm_state: NormState) -> List[str]:
    word_expands = []
    was_quoted, word = unquote(word)
    for transform in (expand_file_word, expand_file_ext,
                      expand_glob, expand_ip_addr, expand_bash_var,
                      expand_slash_word):
        word, new_expands = transform(word, norm_state)
        word_expands.extend(new_expands)
    if was_quoted:
        word_expands.append("WASQUOTEDWORD")
        if " " in word:
            word_expands.append("WASMULTIWORDQUOTE")
        if any(any(subword == util for util in common_utils) for subword in word.split()):
            # If it looks like it is a quoted command, then don't expand
            word = word
        else:
            word = ""
    if word:
        word_expands.append(word)
    return word_expands


def expand_file_word(word: str, norm_state: NormState) -> Tuple[str, List[str]]:
    expands = []
    unquoted_word = word.lower()
    if norm_state.files_seen.has_seen(word) or looks_like_a_file(unquoted_word):
        expands.append(norm_state.files_seen.convert(word))
        if "/" in word:
            expands.append("CONSTDIRLIKE")
        else:
            expands.append("CONSTNOTDIRLIKE")
        if "*" in word:
            expands.append("FNWITHGLOB")
        for ext in common_exts:
            if unquoted_word.endswith("." + ext.lower()):
                if ext in useful_ext:
                    expands.append("EXT" + ext)
                break
    word_left = "" if expands else word.lower()
    return word_left, expands


def expand_slash_word(word: str, norm_state: NormState) -> Tuple[str, List[str]]:
    if word.count("/") == 1:
        a, b = word.split("/")
        if a and b:
            return word, [a, b]
    return word, []


def expand_file_ext(word: str, norm_state: NormState) -> Tuple[str, List[str]]:
    expands = []
    if word.lower() in common_exts:
        expands.append("CONSTFILEEXTENSION")
        if word.lower() in useful_ext:
            expands.append(word)
        word = ""
    return word, expands


def expand_glob(word: str, norm_state: NormState) -> Tuple[str, List[str]]:
    if "*" in word:
        return "", ["WORDWITHGLOB"]
    return word, []


def expand_ip_addr(word: str, norm_state: NormState) -> Tuple[str, List[str]]:
    by_dot = word.split(".")
    if len(by_dot) == 4 and all(s.isnumeric() for s in by_dot):
        return "", ["CONSTIPADDRRESS"]
    return word, []


def expand_bash_var(word: str, norm_state: NormState) -> Tuple[str, List[str]]:
    if word.startswith("$") and all(b not in word for b in ("/", ".", "(", ")", "\\")):
        return "", [norm_state.bash_vars.convert(word)]
    return word, []


def clean_punkt(nl: str) -> str:
    if nl[-1] in (".", "?"):
        nl = nl[:-1]
    return nl


def cheap_replacers(nl: str) -> str:
    for i in range(10):
        fill = str(i) + '=' if i > 0 else ''
        nl = nl.replace(f"[-[{fill}FILENAME]-]", f"cheapreplacer{i}.txt")
        nl = nl.replace(f"[-[{fill}DIRNAME]-]", f"cheapdirnamerepl.txt")
        nl = nl.replace(f"[-[{fill}LETTER]-]", f"a")
        nl = nl.replace(f"[-[{fill}ENGWORD]-]", f"engwordreplacematch")
        nl = nl.replace(f"[-[{fill}USERNAME]-]", f"uname")
        nl = nl.replace(f"[-[{fill}EXTENSION]-]", f"cheapextrepl")
        nl = nl.replace(f"[-[{fill}GROUPNAME]-]", f"cheapgroupname")
    return nl


def replace_files(s: str):
    out_words = []
    files_seen = {}
    for word in s.split():
        was_quoted, unquoted_word = unquote(word)
        unquoted_word = unquoted_word.lower()
        if unquoted_word in files_seen:
            out_words.append(files_seen[unquoted_word])
            continue
        if looks_like_a_file(word):
            norm = f"CONSTPATHNAME{len(files_seen)}"
            files_seen[unquoted_word] = norm
            out_words.append(norm)
            continue
        out_words.append(unquoted_word)
    return " ".join(out_words)


def replace_numbers(s: str):
    def isnum(w: str):
         return w.isdigit()

    return " ".join([
        word if not isnum(word) else "CONSTNUMBER"
        for word in s.split()
    ])


def unquote(s: str):
    for q in ('"', "'"):
        if s.startswith(q) and s.endswith(q):
            return True, s[1:-1]
    return False, s


common_exts = {"sh", "py", "txt",  "png", "tmp",  "zip",
               "csv", "cpp", "h", 'html', 'go', 'sql', 'json', 'java', "tar", "alias",
               "tar.gz", "php", 'js', "ml", "gz", "svn", "ogg"}
useful_ext = {"txt", "tar", "tar.gz", "java", "h", "zip", "py", "sh", "gz"}
common_utils = {"cp", "sed", "mv", "tar", "mv", "rm"}


def looks_like_a_file(string: str):
    """A super hacky heuristic function to guess if a string looks like a path"""
    was_quoted, string = unquote(string)
    if string.lower() in ("filename", "{filename}", "{file}", "[file]", '[filename]]', 'file', '$file'):
        return True
    if string.startswith("s/"):
        # Could be a sed expression. This is admittedly a crappy detection of this...
        return False
    if string.count(".") >= 3 and "/" not in string:
        return False  # Maybe somethign like a ip address?
    non_filey_chars = ("?", "@", "!", "(", ")", "//", "=")
    for c in non_filey_chars:
        if c in string:
            return False
    dot_split = string.split(".")
    if len(dot_split) == 2 and 1 <= len(dot_split[-1]) <= 5:
        return True
    if string.count(".") >= 2 and not string.endswith("."):
        return True
    if string.startswith("./") or string.startswith("../"):
        return True
    if string.count("/") >= 2:
        return True
    if string.startswith("~/") and len(string) > len("~/"):
        return True
    if any(string.endswith("." + ext) for ext in common_exts):
        return True
    if string.startswith("/"):
        return True
    return False
