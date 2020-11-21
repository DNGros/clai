from tqdm import tqdm

from bashlint.data_tools import bash_parser
from exploregivein import cmd_to_seq, maybe_split_short_args, cmd_to_seq_str_edit, strip_var_assign, \
    strip_only_quote, strip_flag_clear
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


def test_dash():
    assert "-prune" in cmd_to_seq_str_edit("find . -name foo -type d -prune -o -name foo -print")


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
    
#def test_perm():
#    assert "find / -perm arg" == cmd_to_seq_str_edit("find / -perm /u+rw,g+r,o+r")
