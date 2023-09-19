import random
import string
from tqdm import tqdm
from data_generation import dictionaries
from functools import cache


consonants = "bcçdfgğhjklmnprsştvyz"

def split_sentence(sentence):
    sentence = sentence.strip()
    sentence = sentence.split(" ")
    return sentence

def flip_coin(probability):
    return random.random() < probability

def normalize(word):
    return word.replace("İ", "i").replace("I", "ı").lower()

def capitalize(word):
    if word:
        if word[0] == "i":
            return "İ" + word[1:]
        elif word[0] == "ı":
            return "I" + word[1:]
        else:
            return word[0].upper() + word[1:]
    else:
        return word
    
def lowercase(word):
    if word:
        if word[0] == "İ":
            return "i" + word[1:]
        elif word[0] == "I":
            return "ı" + word[1:]
        else:
            return word[0].lower() + word[1:]
    else:
        return word

def reduce_token_index(annotation):
    annotation.starting_index -= 1
    annotation.ending_index -= 1

def increase_token_index(annotation):
    annotation.starting_index += 1
    annotation.ending_index += 1

def adjust_tag_indices(tags):
    from annotation import Annotation

    annotations = [Annotation(tag) for tag in tags]

    for i in reversed(range(len(tags))):
            if annotations[i].space_removing: #Reduce one index on previously added tags.
                for j in range(len(tags)):

                    if (annotations[i].ending_index+1) <= annotations[j].starting_index: 
                        reduce_token_index(annotations[j])
            elif annotations[i].space_adding:
                for j in range(len(tags)):
                    if (annotations[i].ending_index-1) <= annotations[j].starting_index:
                        increase_token_index(annotations[j])

    adjusted_tags = []
    for annotation in annotations:
        adjusted_tags.append(annotation.to_string())

    return adjusted_tags

@cache
def get_reverse_dictionaries(rule_number):
    dictionary = getattr(dictionaries, f"dict_rule_{rule_number}")
    return {v: k for k, v in dictionary.items()}

def combine(sentence):
    remove_empty_strings = [word for word in sentence if word]
    return " ".join(remove_empty_strings).strip()

def replace_last(s, old, new):
    li = s.rsplit(old, 1)
    return new.join(li)

def flatten(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]

def contains_space(word):
    return " " in word

def is_capitalized(word):
    return word and word[0].isupper()

def is_lowercase(word):
    return word and word[0].islower()

#Fix apostrophes and quotes
def fix_apostrophes_and_quotes(text):
    text = text.replace("”", '"')
    text = text.replace("“", '"')
    text = text.replace("''", '"')
    text = text.replace("’", "'")
    text = text.replace("‘", "'")
    text = text.replace("´", "'")
    text = text.replace("`", "'")
    return text
