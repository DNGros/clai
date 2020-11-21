import shlex
from typing import Tuple, List, Dict, Iterable
import re

from nltk import PorterStemmer

replacer_re = re.compile(r"\[-\[(\d*=)?\w+\]-\]")

space_splitter_re = re.compile("( \(|\) |\)$|, | |\\\".*?\\\"|'.*?')")

stop_words = {
    "the", "in", "all", "and", "current", "of",
    "to", "with", "for", ",", "a", "that"
}




def normalize_nl(nl: str) -> str:
    _, nl = unquote(nl)
    nl = clean_punkt(nl)
    nl = cheap_replacers(nl)
    nl = word_by_word_expand(nl)
    #nl = replace_files(nl)
    #nl = replace_numbers(nl)
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
    def __init__(self, prefix: str, max_count: int = 3):
        self._conversions = {}
        self._prefix = prefix
        self._max_count = max_count

    def convert(self, val: str):
        if val in self._conversions:
            return self._conversions[val]
        val = len(self._conversions)
        if val > self._max_count:
            val = "MAX"
        new_val = f"{self._prefix}{val}"
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
    for transform in (expand_url, expand_file_word, expand_file_ext, expand_file_size,
                      expand_glob, expand_ip_addr, expand_bash_var,
                      expand_slash_word, expand_number_word, expand_ordinal, expand_dash_word):
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


url_words = ("://", ".com", "net", ".gov", ".edu")


def expand_url(word: str, norm_state: NormState) -> Tuple[str, List[str]]:
    if any(url_word in word for url_word in url_words):
        return "", ["CONSTURL"]
    return word, []


def expand_dash_word(word: str, norm_state: NormState) -> Tuple[str, List[str]]:
    dash_split = word.split("-")
    if len(dash_split) > 1:
        return "", [*dash_split, "WASDASHWORD"]
    return word, []


def expand_number_word(word: str, norm_state: NormState) -> Tuple[str, List[str]]:
    def isnum(w: str):
        return w.isdigit()

    expands = []
    if isnum(word):
        expands.append("CONSTNUMBER")
        if len(word) < 3:
            expands.append("CONSTONEORTWODIGIT")
        if len(word) == 3:
            expands.append("CONSTTHREEDIGIT")
        if len(word) > 3:
            expands.append("CONSTMORETHAN4")
        return "", expands

    return word, []


def expand_file_size(word: str, norm_state: NormState) -> Tuple[str, List[str]]:
    for ext in ('kb', 'mb', 'gb', 'g', 'k', 'm', 'bytes'):
        splited = word.lower().split(ext)
        if len(splited) == 2 and len(splited[1]) == 0 and splited[0].isdigit():
            return "", ["CONSTDATASIZE"]
    return word, []


def expand_file_word(word: str, norm_state: NormState) -> Tuple[str, List[str]]:
    expands = []
    unquoted_word = word.lower()
    if norm_state.files_seen.has_seen(word) or looks_like_a_file(unquoted_word):
        expands.append(norm_state.files_seen.convert(word))
        if "/" in word or 'dir' in word or 'folder' in word:
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


ordinal_set = {
    "first", "1st", "second", "2nd", "third", "3rd", "fourth", "4th", "fifth", "5th",
    "sixth", "6th", "seventh", "7th", "eighth", "8th", "ninth", "9th", "tenth", "10th"
}

def expand_ordinal(word: str, norm_state: NormState) -> Tuple[str, List[str]]:
    if word in ordinal_set:
        return "", ["CONSTORDINAL"]
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

    def const_for_number():
        pass

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
               "tar.gz", "php", 'js', "ml", "gz", "svn", "ogg", 'jpg', 'jpeg', 'txt'}
useful_ext = {"txt", "tar", "tar.gz", "java", "h", "zip", "py", "sh", "gz"}
common_utils = {"cp", "sed", "mv", "tar", "mv", "rm"}
file_name_esq = {"filename", "{filename}", "{file}", "[file]", '[filename]]', '$file', 'dir1', 'file1',
                 'dir2', 'file2', 'folder1', 'folder2', 'somedirectory', "dir/"}


def looks_like_a_file(string: str):
    """A super hacky heuristic function to guess if a string looks like a path"""
    was_quoted, string = unquote(string)
    if string.lower() in file_name_esq:
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
    slash_count = string.count("/")
    has_dot = "." in string
    if slash_count >= 1 and string.endswith("/"):
        return True
    if slash_count >= 2 or (slash_count == 1 and has_dot):
        return True
    if string.startswith("~/") and len(string) > len("~/"):
        return True
    if any(string.endswith("." + ext) for ext in common_exts):
        return True
    if string.startswith("/"):
        return True
    return False


def filter_stop_words(words: Iterable[str]) -> Iterable[str]:
    return (
        word
        for word in words
        if word not in stop_words
    )


stemmer = PorterStemmer()


def stem_words(words: Iterable[str]) -> Iterable[str]:
    return (stemmer.stem(word) for word in words)