import chardet


def detect_encoding(text):
    result = chardet.detect(text)
    return result['encoding']


def convert_to_utf8(text):
    encoding = detect_encoding(text)
    if encoding == 'utf-8':
        # If it's already UTF-8, decode to get the string
        return text.decode('utf-8')
    else:
        # elif encoding == 'ISO-8859-1':  # Latin-1 encoding
        # Decode from Latin-1 and re-encode to UTF-8
        return text.decode(encoding).encode('utf-8').decode('utf-8')
        # return text.decode('ISO-8859-1').encode('utf-8').decode('utf-8')

    # else:
    #     raise ValueError(f"Unexpected encoding: {encoding}")
