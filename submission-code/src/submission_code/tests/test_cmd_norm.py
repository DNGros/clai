from tqdm import tqdm

from bashlint.data_tools import bash_parser
from exploregivein import *
from readdata import get_all_data


#def test_all_cmds_parse():
#    for ex in tqdm(get_all_data(preparse=False).examples):
#        cmd_to_seq(ex.cmd, verify_equal=True)

def test_split_short_args():
    cmd = "ls -la"
    result = maybe_split_short_args(cmd, bash_parser(cmd))
    assert result == "ls -l -a"


def test_split_short_args2():
    cmd = "ls -la | find -name foo -type f"
    result = maybe_split_short_args(cmd, bash_parser(cmd))
    assert result == "ls -l -a | find -name foo -type f"


#def test_dash():
#    assert "-prune" in cmd_to_seq_str_edit("find . -name foo -type d -prune -o -name foo -print")


def test_what():
    assert "-g" not in cmd_to_seq_str_edit('hostname -I | awk \'{print $2}\' | cut -f1,2,3 -d"."')
    assert "-f arg" in cmd_to_seq_str_edit('hostname -I | awk \'{print $2}\' | cut -f1,2,3 -d"."')
    
    
def test_strip_var():
    assert "\"http://$(whoami).$(hostname -f)/path/to/file\"" == \
           strip_var_assign("path=\"http://$(whoami).$(hostname -f)/path/to/file\"")
    
    
def test_strip_quote():
    assert strip_only_quote('"http://$(whoami).$(hostname -f)/path/to/file"') == \
           '"$(whoami).$(hostname -f)"'


def test_strip_flag_clear():
    assert "set $(cal )" == strip_flag_clear("set -- $(cal )")


def test_strip_surroundthing():
    cmd = "cp /lib/modules/$(uname -r)/kernel/drivers/"
    assert "cp $(uname -r)" == cmd_to_seq_str_edit(cmd)


def test_strip_surroundthing2():
    cmd = "grep /lib/modules/`uname -r`/modules.alias"
    assert "grep `uname -r`" == strip_words_with_cmd_sub(cmd)


def test_strip_surroundthing4():
    cmd = "cat /boot/config-`uname -r` | grep"
    assert "cat $(uname -r) | grep" == cmd_to_seq_str_edit(cmd)


def test_strip_surroundthing5():
    cmd = "cat `uname -r`"
    assert "cat $(uname -r)" == replace_backtick_sub_with_dollar_sub(cmd)


def test_clean_paren():
    cmd = "cat $(uname -r )"
    assert "cat $(uname -r)" == cmd_to_seq_str_edit(cmd)


def test_clean_only_sib():
    cmd = "$(uname -r )"
    assert "uname -r" == cmd_to_seq_str_edit(cmd)
    
    
def test_clean_regex():
    cmd = "cat | egrep -i '(\\txt | \\html?)$'"
    assert "cat | egrep -i" == cmd_to_seq_str_edit(cmd)
    
    
def test_clean_regex2():
    cmd = "find -type f -print0 | egrep -i -a -z -Z " \
          "'(\\txt | \\html?)$' | grep -v -a -z -Z | xargs -n arg -0 grep -c -H -i | egrep -v"
    cmd = "find -print0 | egrep -iazZ '(\.txt|\.html?)$'"
    print(cmd_to_seq_str_edit(cmd))
    assert "html" not in cmd_to_seq_str_edit(cmd)
    
    
def test_sed1():
    a = "find -type f -print0 | xargs -0 -n arg md5sum | sort -k arg | uniq -w arg " \
          "-d --all-repeated=arg | sed -e 's^[0-9a-f]*\\ *;"
    cmd = "ind -type f -print0 | xargs -0 uniq -w arg -d | sed -e 's^[0-9a-f]*\\ *;'"
    assert "0-9a-f" not in cmd_to_seq(a)
    
    pass
