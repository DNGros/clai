from datacleaning import normalize_nl, pretokenize


def test_norm1():
    assert "normal " in normalize_nl("all normal/regular")


def test_norm1a():
    assert "screen " in normalize_nl("the screen/window width")
    
    
def test_norm1c():
    norm = normalize_nl("Page through the contents of 'file', but excess from "
                        "lines that don't fit within the screen/window width is cut.")
    print(norm)
    assert "screen " in norm


def test_norm2():
    assert "cp" in normalize_nl("run 'cp foo' command")


def test_normurl():
    assert "CONSTURL" in normalize_nl("run http://example.com command")


def test_normsize():
    assert "CONSTDATASIZE" in normalize_nl("in 40mb size")

def test_dash():
    assert "foo" in normalize_nl("in foo-bar size")
    assert "bar" in normalize_nl("in foo-bar size")
    assert "foo-bar" not in normalize_nl("in foo-bar size")

def test_space_spliiter():
    assert pretokenize("hello world") == ["hello", "world"]

def test_space_spliiter4():
    assert pretokenize("do a 'hello world' message") == ["do", "a", "'hello world'", "message"]
    assert pretokenize("this april, 19th") == ["this", "april", ",", "19th"]

def test_space_spliiter3():
    assert pretokenize("this april,19th") == ["this", "april,19th"]
    #assert pretokenize("please don't break 'here'") == ["please", "don't", 'break', "'here'"]

def test_space_spliiter2():
    assert pretokenize("this (is a test)") == ["this", "(", "is", "a", "test", ")"]


