import itertools
import re
import random
from typing import Iterable, List, TypeVar
from tqdm import tqdm
from collections import Counter

from bashlint import nast
from bashlint.data_tools import bash_parser
from bashlint.nast import CommandSubstitutionNode
from innereval.evaluate import make_preparse, make_preparse_cached
import torchtext
from readdata import get_all_data, DataItem
import attr



@attr.s(auto_attribs=True, frozen=True)
class PredSet:
    utils: List[int]


UNK = "<unk>"


class SetExamplesVectorizer():
    def __init__(
        self,
        examples: Iterable[DataItem],
        min_nl_freq: int = 2,
        min_util_freq: int = 1,
        min_flag_freq: int = 1,
    ):
        all_flags = Counter()
        all_cmds = Counter()
        cmd_lens = Counter()
        word_list = Counter()
        for example in tqdm(examples):
            preparse = make_preparse_cached(example.cmd)
            for flag_set in preparse.flag_nodes:
                all_flags.update(flag_set)
            all_cmds.update(preparse.utility_names)
            cmd_lens.update([len(preparse.utility_names)])
            word_list.update(example.nl_norm.split())
        self.vocab = torchtext.vocab.Vocab(word_list, min_freq=min_nl_freq)
        self.util_to_ind = {UNK: 0, '<null>': 1}
        for util, count in all_cmds.items():
            if util not in self.util_to_ind and count >= min_util_freq:
                self.util_to_ind[util] = len(self.util_to_ind)
        self.flag_to_ind = {UNK: 0}
        for flag, count in all_flags.items():
            if flag not in self.flag_to_ind and count >= min_flag_freq:
                self.util_to_ind[flag] = len(self.util_to_ind)

    def nl_to_inds(self, tokens):
        return self.vocab.lookup_indices(tokens)

    def convert_pred(self, cmd: str):
        preparse = make_preparse_cached(cmd)
        assert len(preparse.utility_names) >= 1, cmd
        unk_ind = self.util_to_ind[UNK]
        return PredSet(
            [self.util_to_ind.get(util, unk_ind) for util in preparse.utility_names]
        )

    def nl_pad_ind(self) -> int:
        return self.vocab.lookup_indices(["<pad>"])[0]

    def nl_vocab_size(self) -> int:
        return len(self.vocab)

    def num_utils(self) -> int:
        return len(self.util_to_ind)


def get_all_flags():
    all_flags = Counter()
    all_cmds = Counter()
    cmd_lens = Counter()
    word_list = Counter()
    word_lens = []
    examples = get_all_data(preparse=False).examples
    for example in tqdm(examples):
        preparse = make_preparse_cached(example.cmd)
        for flag_set in preparse.flag_nodes:
            all_flags.update(flag_set)
        all_cmds.update(preparse.utility_names)
        cmd_lens.update([len(preparse.utility_names)])
        #if len(preparse.utility_names) >= 5:
        #    print(example)
        word_list.update(example.nl_norm.split())
        word_lens.append(len(example.nl_norm.split()))
    print(all_flags)
    print(len(all_flags))
    print(all_cmds)
    print(len(all_cmds))
    print(cmd_lens)
    print("word list", word_list)
    print(len(word_list))
    print(max(word_lens))
    print("rsync count", all_cmds['rsync'])


def toy_ast():
    ast = bash_parser("find -type f -name foo | echo bar")


flag_map = {
    (None, "-exec::;"): ("-exec", None, r"\;"),
    (None, "-exec::+"): ("-exec", None, r"+"),
    (None, "-execdir::;"): ("-execdir", None, r"\;"),
    (None, "-execdir::+"): ("-execdir", None, r"+"),
    (None, "-ok::;"): ("-okdir", None, r"\;"),
    (None, "-okdir::;"): ("-okdir", None, r"\;"),
    #('rsync', "-e"): ('-e', "'", None, r"'"),
}

cmd_map = {

}


def cmd_to_seq(cmd: str, verify_equal: bool = False):
    ast = bash_parser(cmd)

    def explore(node: nast) -> List[str]:
        out = []

        def visit_all_children():
            if len(node.children) > 1 and all(child.is_utility() or child.kind == "pipeline"
                                              for child in node.get_children()):
                out.append('"')
                def make_wrapper(child):
                    w = CommandSubstitutionNode(parent=node)
                    w.children = [child]
                    return w
                out.extend(flatten_list(
                    explore(make_wrapper(child))
                    for child in node.get_children())
                )
                out.append('"')
            else:
                out.extend(flatten_list(explore(child) for child in node.get_children()))

        #print("NODE", node.kind, node.value, node.is_reserved())
        if node.is_root():
            visit_all_children()
        elif node.kind == "pipeline":
            for child in node.get_children():
                out.extend(explore(child))
                out.append("|")
            out.pop()  # Remove hanging pipe
        elif node.kind == "commandsubstitution":
            children = list(flatten_list(explore(child) for child in node.get_children()))
            children[0] = "$(" + children[0]
            children[-1] += ")"
            out.extend(children)
            #out.append("$(")
            #out.extend(flatten_list(explore(child) for child in node.get_children()))
            #out.append(")")
        elif node.kind == "processsubstitution":
            #out.append("<(")
            children = list(flatten_list(explore(child) for child in node.get_children()))
            children[0] = "<(" + children[0]
            children[-1] += ")"
            out.extend(children)
            #out.append(")")
        elif node.is_utility():
            out.append(node.value)
            out.extend(flatten_list(explore(child) for child in node.get_children()))
        elif node.kind == "t":
            # Weird thing inside a commandsub?
            out.extend(flatten_list(explore(child) for child in node.get_children()))
        elif node.is_option():
            #print("OPTION", node.value, node.parent)
            for k in ((node.parent.value, node.value), (None, node.value)):
                if k in flag_map:
                    for replace in flag_map[k]:
                        if replace is None:
                            out.extend(flatten_list(explore(child) for child in node.get_children()))
                            #visit_all_children()
                        else:
                            out.append(replace)
                    break
            else:
                out.append(node.value)
            if any(
                    child.is_utility() or child.kind == "pipeline"
                    for child in node.get_children()
            ):
                out.append("'")
                out.extend(flatten_list(explore(child) for child in node.get_children()))
                out.append("'")
            else:
                out.extend(flatten_list(explore(child) for child in node.get_children()))
                #visit_all_children()
        elif node.is_argument():
            #print("ARG", node.kind, node.value, node.get_children(), node.parent.kind, node.parent.value)
            if node.parent.kind == "commandsubstitution":
                out.append('~')
            elif node.parent.value == "ssh":
                out.append(node.value)
            else:
                out.append("arg")
        elif node.kind == "bracket":
            # Find specific
            out.append(r"\(")
            out.extend(flatten_list(explore(child) for child in node.get_children()))
            out.append(r"\)")
        elif node.kind == "binarylogicop":
            out.append("-or")
        elif node.kind == "unarylogicop":
            if(node.value == '!'):
                out.append('\!')
            else:
                out.append(node.value)
        elif node.kind == "operator":
            if(node.value == '!'):
                out.append('\!')
            else:
                out.append(node.value)
        else:
            raise ValueError(node.kind, node.value, cmd, node.children)

        return out

    out = explore(ast)
    norm_cmd = " ".join(out)
    if verify_equal:
        old_preparse = make_preparse_cached(cmd)
        new_preparse = make_preparse_cached(norm_cmd)
        if old_preparse != new_preparse:
            print("Old")
            print(cmd)
            print(old_preparse)
            print("new")
            print(norm_cmd)
            print(new_preparse)
            if "rsync" in norm_cmd:
                print("rsync weird")
                #if old_preparse.utility_names != new_preparse.utility_names:
                #raise ValueError
            else:
                raise ValueError

    return norm_cmd


def is_same_preparse(gt: str, canidate: str, exception_if_not_eq: bool = False) -> bool:
    old_preparse = make_preparse_cached(gt)
    new_preparse = make_preparse_cached(canidate)
    are_equal = old_preparse == new_preparse
    if exception_if_not_eq and not are_equal:
        print("Old")
        print(gt)
        print(old_preparse)
        print("new")
        print(canidate)
        print(new_preparse)
        raise ValueError
    return are_equal


def find_maybe_constants(node: nast, cmd) -> List[str]:
    out = []
    if node.is_argument():
        if node.value in cmd:
            out.append(node.value)
        #if "*" in node.value and '.' in node.value:
        #    # Weird thing where in fine '*java' gets expanded into '*.java'
        #    out.append(node.value.replace('.', ''))
    out.extend(flatten_list(find_maybe_constants(child, cmd) for child in node.get_children()))
    return out


def find_maybe_flags(node: nast, cmd) -> List[str]:
    out = []
    if node.is_option():
        out.append(node.value)
    elif node.kind == "binarylogicop" or node.kind == "unarylogicop" or node.kind == "operator":
        out.append(node.value)

    out.extend(flatten_list(find_maybe_flags(child, cmd) for child in node.get_children()))
    return out


short_args_re = re.compile(r' -[A-Za-z]+')


def maybe_split_short_args(cmd: str, ast: nast) -> str:
    maybe_flags = find_maybe_flags(ast, cmd)
    flags_no_dash = {flag[1:] for flag in maybe_flags}
    new_cmd = cmd
    # Try and split short args
    for maybe_short_args in short_args_re.finditer(cmd):
        # match in the form " -lsa"
        match = maybe_short_args.group(0)
        has_arg_at_end = maybe_short_args.group(0).endswith("arg")
        if has_arg_at_end:
            match = match[:len("arg")]
        args = match[len(" -"):]
        if args in flags_no_dash and not has_arg_at_end:
            continue  # Either already a single arg, or can't be split (like "-name")
        split = " -" + " -".join(args)
        if has_arg_at_end:
            split += " arg"
        maybe_new = new_cmd.replace(maybe_short_args.group(0), split)
        if is_same_preparse(cmd, maybe_new):
            new_cmd = maybe_new
    return new_cmd


quote = ('"', "'")

def strip_var_assign(cmd: str) -> str:
    eq_split = cmd.split("=")
    if len(eq_split) == 2:
        if len(eq_split[0].split()) == 1 and eq_split[1][0] in quote and eq_split[1][0] == eq_split[1][-1]:
            new_maybe = eq_split[1]
            if is_same_preparse(cmd, new_maybe):
                return new_maybe
    return cmd


cmd_sub_strip_re = re.compile(r"(\S*)(\$\([^\)]+\)*\)|\`[^\`]+\`*\`)(\S*)")


def strip_words_with_cmd_sub(cmd: str) -> str:
    new_cmd = cmd
    for match in cmd_sub_strip_re.finditer(cmd):
        before_sub, sub, after_sub = match.groups()
        if not before_sub and not after_sub:
            continue  # Nothing to replace
        maybe_new = cmd.replace(match.group(0), sub)
        if is_same_preparse(cmd, maybe_new):
            new_cmd = maybe_new
    return new_cmd


backtick_sub_re = re.compile(r"\`.+?\`")


def replace_backtick_sub_with_dollar_sub(cmd):
    new_cmd = cmd
    for match in cmd_sub_strip_re.finditer(cmd):
        maybe_new = cmd.replace(match.group(0), '$(' + match.group(0)[1:-1] + ")").strip()
        if is_same_preparse(cmd, maybe_new):
            new_cmd = maybe_new
    return new_cmd


def try_clean_paren_space_before_paren(cmd: str) -> str:
    maybe_new = cmd.replace(" )", ")").strip()
    if maybe_new != cmd and is_same_preparse(cmd, maybe_new):
        return maybe_new
    return cmd


def try_unpack_only_command_sub(cmd) -> str:
    if cmd.startswith("$(") and cmd.endswith(")"):
        maybe_new = cmd[len("$("):-len(")")].strip()
        if maybe_new != cmd and is_same_preparse(cmd, maybe_new):
            return maybe_new
    return cmd



def safe_str_index(s: str, search):
    try:
        return s.index(search)
    except ValueError:
        return len(s)


def safe_str_rindex(s: str, search):
    try:
        return s.rindex(search)
    except ValueError:
        return 0


def strip_only_quote(cmd: str) -> str:
    if cmd[0] in quote and cmd[0] == cmd[-1] and ("$" in cmd or "`" in cmd):
        new_maybe = cmd[min(safe_str_index(cmd, "$"), safe_str_index(cmd, '`')):
                        max(safe_str_rindex(cmd, ')') + 1, safe_str_rindex(cmd, '`') + 1)]
        new_maybe = cmd[0] + new_maybe + cmd[-1]
        if is_same_preparse(cmd, new_maybe):
            return new_maybe
    return cmd


def strip_flag_clear(cmd):
    if " -- " in cmd:
        new_maybe = cmd.replace("-- ", "")
        if is_same_preparse(cmd, new_maybe):
            return new_maybe
    return cmd


def try_strip_every_word(cmd):
    new_cmd = cmd
    for word in cmd.split():
        maybe_new = remove_extra_whitespace(new_cmd.replace(word, ""))
        if is_same_preparse(cmd, maybe_new):
            new_cmd = maybe_new
            continue
        maybe_new = remove_extra_whitespace(new_cmd.replace(word, "arg"))
        if is_same_preparse(cmd, maybe_new):
            new_cmd = maybe_new
            continue
    return new_cmd


def remove_extra_whitespace(s: str) -> str:
    return " ".join(t for t in s.split() if t)


def cmd_to_seq_str_edit(cmd: str):
    ast = bash_parser(cmd)
    maybe_constants = find_maybe_constants(ast, cmd)

    norm_cmd = cmd

    # Try and remove out constants
    for const in maybe_constants:
        for const in (const, const.replace(".", "")):
            # First try to just get rid of it
            maybe_new = remove_extra_whitespace(norm_cmd.replace(const, ""))
            if is_same_preparse(cmd, maybe_new):
                norm_cmd = maybe_new
                continue
            # Try and replace with an arg
            repl_val = "arg"
            maybe_new = norm_cmd.replace(const, "arg")
            if is_same_preparse(cmd, maybe_new):
                norm_cmd = maybe_new
                continue

    # Sometimes get something like "ls -l|wc". Split the pipe so more consistent
    if '|' in norm_cmd:
        maybe_new = " ".join(t for t in norm_cmd.replace('|', " | ").split() if t)
        if is_same_preparse(cmd, maybe_new):
            norm_cmd = maybe_new

    norm_cmd = maybe_split_short_args(norm_cmd, ast)
    norm_cmd = strip_var_assign(norm_cmd)
    norm_cmd = strip_only_quote(norm_cmd)
    norm_cmd = strip_flag_clear(norm_cmd)
    norm_cmd = strip_words_with_cmd_sub(norm_cmd)
    norm_cmd = replace_backtick_sub_with_dollar_sub(norm_cmd)
    norm_cmd = try_clean_paren_space_before_paren(norm_cmd)
    norm_cmd = try_unpack_only_command_sub(norm_cmd)
    norm_cmd = try_strip_every_word(norm_cmd)
    norm_cmd = norm_cmd.strip()

    is_same_preparse(cmd, norm_cmd, exception_if_not_eq=True)

    return norm_cmd



T = TypeVar('T')
def flatten_list(l: Iterable[Iterable[T]]) -> List[T]:
    return [item for sublist in l for item in sublist]

    #out_str = cmd

    #def find_args(node: nast):
    #    if node.is_argument():
    #        print("ARG", node.value)
    #    for child in node.get_children():
    #        find_args(child)

    #find_args(ast)
    #return out_str


def main():
    #print(make_preparse('find / -name "*.core" -print -exec rm {} \;'))
    #print(make_preparse("find -type f -name $(echo -n 'foo')"))
    #print(make_preparse(r"find . -name \*.xml | grep -v /workspace/ | tr '\n' '\0' | xargs -0 tar -cf xml.tar"))
    #print(make_preparse(r"find . -name \*.xml -o -name foio"))
    #get_all_flags()
    #print('TEST', make_preparse(r'xargs ls -l'))
    #get_all_flags()
    #print("parse", cmd_to_seq("ls -l foo | wc"))
    #print("parse", cmd_to_seq('find / -name "*.core" -print -exec rm {} \;', verify_equal=True))
    #print("parse", cmd_to_seq('cat $( uname -r ) | cat foo', verify_equal=True))
    #print("parse", cmd_to_seq('PROMPT_COMMAND=\'e "$(date foo) $(history 1 |cut -c 7-)" >> /tmp/trace\'', verify_equal=True))
    #print("parse", cmd_to_seq('rsync --progress -avhe ssh arg arg', verify_equal=True))
    #print("parse", cmd_to_seq_str_edit('cd $( ~/foo sdf sdf )', verify_equal=True))
    cmd = "cut -b8-"
    print(maybe_split_short_args(cmd, bash_parser(cmd)))
    print(make_preparse("find / -perm /u+rw,g+r,o+r"))
    for ex in random.sample(get_all_data(False).examples, k=20):
        print("---")
        print("before", ex.cmd)
        print("parse", cmd_to_seq_str_edit(ex.cmd))


    #toy_ast()


if __name__ == "__main__":
    main()
