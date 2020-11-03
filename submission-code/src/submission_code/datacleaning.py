from typing import Tuple
import re
replacer_re = re.compile(r"\[-\[(\d*=)?\w+\]-\]")


def normalize_nl(nl: str) -> str:
    return replace_numbers(replace_files(cheap_replacers(nl)))


def replace_cmd(cmd: str) -> str:
    return re.sub(replacer_re, "replval", cmd)



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
        was_quoted, unquoted_word = unqote(word)
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



def unqote(s: str):
    for q in ('"', "'"):
        if s.startswith(q) and s.endswith(q):
            return True, s[1:-1]
    return False, s



common_exts = (".sh", ".py", ".txt", ".gz", ".png", ".tmp", ".xml", ".zip",
               ".csv", ".cpp", ".h", 'html', '.go', '.sql', '.json', '.java', ".tar", "alias")


def looks_like_a_file(string: str):
    """A super hacky heuristic function to guess if a string looks like a path"""
    was_quoted, string = unqote(string)
    if string.lower() in ("filename", "{filename}", "{file}", "[file]", '[filename]]', 'file', '$file'):
        return True
    if string.startswith("s/"):
        # Could be a sed expression. This is admittedly a crappy detection of this...
        return False
    non_filey_chars = ("?", "@", "!", "(", ")", "//", "=")
    for c in non_filey_chars:
        if c in string:
            return False
    if string.endswith(common_exts):
        return True
    dot_split = string.split(".")
    if len(dot_split) == 2 and 1 <= len(dot_split[-1]) <= 5:
        return True
    if string.count(".") >= 2 and not string.endswith("."):
        return True
    if string.startswith("./") or string.startswith("../"):
        return True
    if string.count("/") >= 2:
        return True
    return False
