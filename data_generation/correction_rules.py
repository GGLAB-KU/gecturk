
from data_generation.utils import *


def correct_rule_1(sentence, tags, index):
    """Conjunction de/da must written separately. """
    word = normalize(sentence[index])

    if "de" in word and "da" in word:
        suffix_index = max(word.rfind("de"), word.rfind("da"))
    elif "da" in word:
        suffix_index = word.rfind("da")
    elif "de" in word:
        suffix_index = word.rfind("de")
    else:
        raise Exception("Conjunction de/da must be in the word.")
        
    sentence[index] = sentence[index][:suffix_index] + " " + sentence[index][suffix_index:]
    tags[index] = 0

def correct_rule_2(sentence, tags, index):
    """Conjunction de/da must obey Vowel Harmony Rules (Büyük Ünlü Uyumu)."""

    if sentence[index] == "de": 
        sentence[index] = "da"
    elif sentence[index] == "da": 
        sentence[index] = "de"

    tags[index] = 0

def correct_rule_3(sentence, tags, index):
    """Conjunction de/da should not be spelled as te/ta."""

    if sentence[index] == "te": 
        sentence[index] = "de"
    elif sentence[index] == "ta": 
        sentence[index] = "da"

    tags[index] = 0

def correct_rule_4(sentence, tags, index):
    """"Conjunction da should be written separately when it is used with a ya. (i.e. "ya da")"""

    sentence[index] = sentence[index].replace("yada", "ya da")
    tags[index] = 0

def correct_rule_5(sentence, tags, index):
    """Conjunction de/da can't be used with an apostrophe. (e.g Ayşe de geldi. (not Ayşe'de geldi.))"""

    sentence[index] = sentence[index].replace("'", " ")
    
    tags[index] = 0

def correct_rule_6(sentence, tags, index):
    """Suffix de/da must be written jointly."""  
    apostrophe = "'" if sentence[index].replace(".", "").isdigit() or (index !=0 and is_capitalized(sentence[index])) else ""
    sentence[index] = sentence[index] + apostrophe + sentence[index+1]
    sentence[index+1] = ""
    tags[index] = 0
    tags[index+1] = 0
    
def correct_rule_7(sentence, tags, index):
    """ Conjunction ki must written separately. """
    word = normalize(sentence[index])
    suffix_index = word.rfind("ki")
    sentence[index] = sentence[index][:suffix_index] + " " + sentence[index][suffix_index:]
    tags[index] = 0

def correct_rule_9(sentence, tags, index):
    """Remove the vowel between two consonants in the beginning of a word."""
    sentence[index] = sentence[index][0] + sentence[index][2:]
    tags[index] = 0

def correct_rule_10(sentence, tags, index):
    """Duplicats must be written separately."""
    word = sentence[index]
    word = word[:int(len(word)/2)] + " " + word[int(len(word)/2):]    
    sentence[index] = word
    tags[index] = 0

def correct_rule_11(sentence, tags, index):
    """Mi/mı/mu/mü should be written separately."""
    word = sentence[index]
    suffix_index = None

    #First check for the suffixes that has multiple occurence of mi/mı/mu/mü
    for suffix in ["miymiş", "mıymış", "muymuş", "müymüş"]:
        if suffix in word:
            suffix_index = word.rfind(suffix)
            break

    if suffix_index is None:
        #If there is no such suffix, check for the last occurence of mi/mı/mu/mü
        for i in reversed(range(0, len(word)-1)):
            if word[i: i+2].lower() in ["mi", "mı", "mu", "mü"]:
                suffix_index = i
                break

    sentence[index] = word[:suffix_index] + " " + word[suffix_index:]
    tags[index] = 0

def correct_rule_12(sentence, tags, index):
    """Suffixes starting with -e/-a are written as is unlike the pronunciation."""

    word = sentence[index]
    reverse_dict = get_reverse_dictionaries(12)

    for wrong, correct in reverse_dict.items():
        if wrong in word:
            break
    
    sentence[index] = word.replace(wrong, correct)
    tags[index] = 0

def correct_rule_20(sentence, tags, index):
    """Conjunction words created with bilmek, vermek, kalmak, durmak, gelmek and yazmak are written jointly if they have one of the following suffixes:
    -a, -e, -ı, -i, -u, -ü
    
    Example:
    yapıvermek, not yapı vermek
    uyuyakalmak, not uyuya kalmak
    """

    sentence[index] = sentence[index] + sentence[index+1]
    sentence[index+1] = ""
    tags[index] = 0
    tags[index+1] = 0

def correct_rule_21(sentence, tags, index):
    """Conjunction words created with hane, name and zade are written jointly."""

    sentence[index] = sentence[index] + sentence[index+1]
    sentence[index+1] = ""
    tags[index] = 0
    tags[index+1] = 0

def correct_rule_23(sentence, tags, index):
    """First word of a sentence must be capitalized."""

    sentence[index] = capitalize(sentence[index])
    tags[index] = 0

def correct_with_dictionary(rule_name, adds_space=False):
    """Get the corrector function that operates on the relevant dictionary.
    Args: 
    rule_name: The name of the rule.
    adds_space: Whether the rule adds space or not.

    Returns: function
    """
    reverse_dict = get_reverse_dictionaries(rule_name)

    def correction_function(sentence, tags, index):
        for wrong, correct in reverse_dict.items():
            if adds_space:
                current = normalize(sentence[index]) + " " + normalize(sentence[index+1])
                if wrong in current:
                    new = current.replace(wrong, correct)
                    sentence[index] = capitalize(new) if is_capitalized(sentence[index]) else new
                    sentence[index+1] = ""
                    tags[index] = 0
                    tags[index+1] = 0
                    return
         
            else:
                current = normalize(sentence[index])
                wrong = normalize(wrong)
                if wrong in current:
                    new = current.replace(wrong, correct)
                    sentence[index] = capitalize(new) if is_capitalized(sentence[index]) else new
                    tags[index] = 0
                    return

    return correction_function


correction_rules = {
    "rule_1": correct_rule_1,
    "rule_2": correct_rule_2,
    "rule_3": correct_rule_3,
    "rule_4": correct_rule_4,
    "rule_5": correct_rule_5,
    "rule_6": correct_rule_6,
    "rule_7": correct_rule_7,
    "rule_8": correct_with_dictionary(8, adds_space=True),
    "rule_9": correct_rule_9,
    "rule_10": correct_rule_10,
    "rule_11": correct_rule_11,
    "rule_12": correct_rule_12,
    "rule_13": correct_with_dictionary(13),
    "rule_14": correct_with_dictionary(14),
    "rule_15": correct_with_dictionary(15),
    "rule_16": correct_with_dictionary(16),
    "rule_17": correct_with_dictionary(17),
    "rule_18": correct_with_dictionary(18, adds_space=True),
    "rule_19": correct_with_dictionary(19, adds_space=True),
    "rule_20": correct_rule_20,
    "rule_21": correct_rule_21,
    "rule_22": correct_with_dictionary(22, adds_space=True),
    "rule_23": correct_rule_23,
    "rule_24": correct_with_dictionary(24),
    "rule_25": correct_with_dictionary(25),
}