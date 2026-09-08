from string_utils import reverse_text, is_palindrome


def test_reverse_text():
    assert reverse_text("hello") == "olleh"


def test_palindrome():
    assert is_palindrome("Level") is True
