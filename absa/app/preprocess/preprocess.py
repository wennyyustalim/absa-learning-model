import nltk
import re
import string
from contractions import CONTRACTION_MAP

class Preprocess(object):

    @staticmethod
    def split_text(text):
        tokenizer = nltk.PunktSentenceTokenizer()
        return tokenizer.tokenize(text)

    @staticmethod
    def expand_contractions(text):
        pattern = re.compile("({})".format("|".join(CONTRACTION_MAP.keys())),flags = re.DOTALL| re.IGNORECASE)

        def replace_text(t):
            txt = t.group(0)
            if txt.lower() in CONTRACTION_MAP.keys():
                return CONTRACTION_MAP[txt.lower()]

        expand_text = pattern.sub(replace_text,text)
        return expand_text

    @staticmethod
    def remove_special_characters(text):
        pattern = re.compile("[{}]".format(re.escape(string.punctuation)))
        return pattern.sub('', text)
