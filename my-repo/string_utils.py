def reverse_text(text):
    return text[::-1]


def is_palindrome(text):
    cleaned = text.lower()
    return cleaned == cleaned[::-1]
