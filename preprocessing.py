import re

from pandas import DataFrame


def text_col(x_NC):
    if isinstance(x_NC, DataFrame):
        return x_NC["text"]
    return [text for website, text in x_NC]


def num_del(x_NC):
    return [re.sub(r"[0-9]", "", text) for text in text_col(x_NC)]


def num_to_sp(x_NC):
    return [re.sub(r"[0-9]", " ", text) for text in text_col(x_NC)]
