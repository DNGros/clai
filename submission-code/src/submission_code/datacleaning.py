from typing import Tuple


def normalize_nl(nl: str) -> str:
    return replace_files(nl)


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



def unqote(s: str):
    for q in ('"', "'"):
        if s.startswith(q) and s.endswith(q):
            return True, s[1:-1]
    return False, s



common_exts = (".sh", ".py", ".txt", ".gz", ".png", ".tmp", ".xml", ".zip",
               ".csv", ".cpp", ".h", 'html', '.go', '.sql', '.json', '.java', ".tar")


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
    if len(dot_split) == 2 and 1 <= len(dot_split[-1]) <= 4:
        return True
    if string.count(".") >= 2 and not string.endswith("."):
        return True
    if string.startswith("./") or string.startswith("../"):
        return True
    if string.count("/") >= 2:
        return True
    return False
