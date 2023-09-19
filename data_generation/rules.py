from utils import *
import dictionaries
import re

def rule_1(sentence, morp_analysis, flags, p, name):
    """Conjunction de/da must written separately. """

    tags = []
    for i in range(len(sentence)):
        word = normalize(sentence[i])

        if flags[i] and (word == "de" or word == "da") and i != 0 and flags[i-1] and flip_coin(p):
            tags.append(f"A {i-1} {i}|||{name}|||{sentence[i-1]} {sentence[i]}|||REQUIRED|||-NONE-|||0")
            sentence[i-1] = sentence[i-1] + sentence[i] 
            
            sentence[i] = ""
            flags[i] = False
            flags[i-1] = False
    return tags
#  Onlar da geldi
#A 0 1|||name|||Onlar da
#S Onlarda geldi.

def rule_2(sentence, morp_analysis, flags, p, name):
    """Conjunction de/da must obey Vowel Harmony Rules (Büyük Ünlü Uyumu)."""


    tags = []
    for i in range(len(sentence)):
        word = normalize(sentence[i])
        
        if flags[i] and (word == "de") and flip_coin(p):
            tags.append(f"A {i} {i+1}|||{name}|||{sentence[i]}|||REQUIRED|||-NONE-|||0")
            sentence[i] = sentence[i].replace("de", "da")
            flags[i] = False
        elif flags[i] and (word == "da") and flip_coin(p):
            tags.append(f"A {i} {i+1}|||{name}|||{sentence[i]}|||REQUIRED|||-NONE-|||0")
            sentence[i] = sentence[i].replace("da", "de")
            flags[i] = False

    return tags

def rule_3(sentence, morp_analysis, flags, p, name):
    """Conjunction de/da should not be spelled as te/ta."""

    tags = []
    for i in range(len(sentence)):
        word = normalize(sentence[i])

        if flags[i] and (word == "de") and flip_coin(p): 
            tags.append(f"A {i} {i+1}|||{name}|||{sentence[i]}|||REQUIRED|||-NONE-|||0")
            sentence[i] = sentence[i].replace("de", "te")
            flags[i] = False
        elif flags[i] and (word == "da") and flip_coin(p):
            tags.append(f"A {i} {i+1}|||{name}|||{sentence[i]}|||REQUIRED|||-NONE-|||0")
            sentence[i] = sentence[i].replace("da", "ta") 
            flags[i] = False

    return tags

def rule_4(sentence, morp_analysis, flags, p, name):
    """"Conjunction da should be written separately when it is used with a ya. (i.e. "ya da")"""
    
    tags = []
    if ("ya" in sentence) and ("da" in sentence) and ((sentence.index("da") - sentence.index("ya")) == 1) and flags[sentence.index("ya")] and flags[sentence.index("da")] and flip_coin(p):
        tags.append(f"A {sentence.index('ya')} {sentence.index('ya') + 1}|||{name}|||ya da|||REQUIRED|||-NONE-|||0")
        flags[sentence.index("ya")] = False
        flags[sentence.index("da")] = False
        sentence[sentence.index("ya")] = "yada"
        sentence[sentence.index("da")] = ""


    return tags
#  Sinemeya ya da tiyatroya gidelim.
#A 1 2|||ya da
#S Sinemaya yada tiyatroya gidelim.

def rule_5(sentence, morp_analysis, flags, p, name):
    """Conjunction de/da can't be used with an apostrophe. (e.g Ayşe de geldi. (not Ayşe'de geldi.))"""

    tags = []
    for i in range(len(sentence)):
        word = normalize(sentence[i])

        if flags[i] and (word == "de" or word == "da") and i != 0 and flags[i-1] and (not "'" in sentence[i-1]) and flip_coin(p):
            tags.append(f"A {i-1} {i}|||{name}|||{sentence[i-1]} {sentence[i]}|||REQUIRED|||-NONE-|||0")
            sentence[i-1] = sentence[i-1] + "\'"+ sentence[i] 
            sentence[i] = ""
            flags[i] = False
            flags[i-1] = False

    return tags
#  Ayşe de geldi.
#A 0 1|||Ayşe de
#S Ayşe'de geldi.

def rule_6(sentence, morp_analysis, flags, p, name):   
    """Suffix de/da must be written jointly."""  

    tags = []
    for i in range(len(sentence)):
        word = normalize(sentence[i])

        if flags[i] and (not "\"" in sentence[i]) and \
        ((word != "de" and word.endswith("de")) or (word != "da" and word.endswith("da"))) and \
        (not "'" in sentence[i] or (i != 0 and is_capitalized(sentence[i]) and ("'de" in sentence[i] or "'da" in sentence[i]))) and \
        ("Loc" in morp_analysis()[i][3]) and flip_coin(p):
            
            tags.append(f"A {i} {i+2}|||{name}|||{sentence[i]}|||REQUIRED|||-NONE-|||0")
            suffix_index = word.rindex("de") if (word != "de" and word.endswith("de")) else word.rindex("da")
            sentence[i] = sentence[i][:suffix_index] + " " + sentence[i][suffix_index:]
            sentence[i] = sentence[i].replace("' ", " ") #E.g. "1962 'de" -> "1962 de"
            flags[i] = False

    return tags
#  Kitapda böyle yazmıyordu.
#A 0 2||Kitapda
#S Kitap da böyle yazmıyordu.

def rule_7(sentence, morp_analysis, flags, p, name):
    """Conjunction ki must written separately."""

    
    tags = []
    for i in range(len(sentence)):
        word = normalize(sentence[i])

        if flags[i] and word == "ki" and i != 0 and flags[i-1] and flip_coin(p):
            tags.append(f"A {i-1} {i}|||{name}|||{sentence[i-1]} {sentence[i]}|||REQUIRED|||-NONE-|||0")
            sentence[i-1] = sentence[i-1] + sentence[i] 
            sentence[i] = ""
            flags[i] = False
            flags[i-1] = False

    return tags

def rule_8(sentence, morp_analysis, flags, p, name):
    """Conjunction ki must written separately. However, there are some exceptions to this rule."""

    exceptions = dictionaries.dict_rule_8
    

    tags = []
    for i in range(len(sentence)):
        word = normalize(sentence[i])

        if flags[i] and word in exceptions and flip_coin(p):
            tags.append(f"A {i} {i+2}|||{name}|||{sentence[i]}|||REQUIRED|||-NONE-|||0")
            word = exceptions[word]
            word = capitalize(word) if is_capitalized(sentence[i]) else word
            
            sentence[i] = word
            flags[i] = False

    return tags
    
def rule_9(sentence, morp_analysis, flags, p, name):
    """Western-origined words that start with double consonants are written without a vowel between the consonants."""

    tags = []
    for i in range(len(sentence)):
        word = normalize(sentence[i])

        if flags[i] and len(word) >= 4 and (word[0] in consonants) and (word[1] in consonants) and (not is_capitalized(word.split("'")[0])) and flip_coin(p):
            tags.append(f"A {i} {i+1}|||{name}|||{sentence[i]}|||REQUIRED|||-NONE-|||0")
           
            sentence[i] = sentence[i][0] + "ı" + sentence[i][1:]
            flags[i] = False
    return tags

def rule_10(sentence, morp_analysis, flags, p, name):
    """Duplicates must be written separately."""


    tags = []
    for i in range(len(sentence)-1):
        if flags[i] and flags[i+1] and (normalize(sentence[i]) == normalize(sentence[i+1])) and flip_coin(p):
            tags.append(f"A {i} {i+1}|||{name}|||{sentence[i]} {sentence[i+1]}|||REQUIRED|||-NONE-|||0")
            
            sentence[i] = sentence[i] + sentence[i+1] 
            sentence[i+1] = ""
            flags[i] = False
            flags[i+1] = False
        
    return tags
    
def rule_11(sentence, morp_analysis, flags, p, name):
    """Mi/mı/mu/mü should be written separately."""


    tags = []
    for i in range(len(sentence)):
        word = normalize(sentence[i])

        if flags[i] and (word.startswith("mu") or word.startswith("mü") or word.startswith("mi") or word.startswith("mı")) and i != 0 and flags[i-1] and ("Ques" in morp_analysis()[i][3]) and flip_coin(p):
            tags.append(f"A {i-1} {i}|||{name}|||{sentence[i-1]} {sentence[i]}|||REQUIRED|||-NONE-|||0")
            sentence[i-1] = sentence[i-1] + sentence[i] 
            sentence[i] = ""
            flags[i] = False
            flags[i-1] = False

    return tags

def rule_12(sentence, morp_analysis, flags, p, name):
    """Suffixes starting with -e/-a are written as is unlike the pronunciation.
    Example: başlayacağım, not başlıyacağım
    """
    mapping = dictionaries.dict_rule_12
    
    tags = []
    for i in range(len(sentence)):
        word = normalize(sentence[i])
        
        if flags[i]:

            for key, value in mapping.items():

                if (key in word) and (word.index(key) > 3) and (not value in word) and ("Verb" in morp_analysis()[i][3]) and flip_coin(p):
                    tags.append(f"A {i} {i+1}|||{name}|||{sentence[i]}|||REQUIRED|||-NONE-|||0")
                    sentence[i] = sentence[i].replace(key, value)

                    flags[i] = False
                    break
                
    return tags

def rule_13(sentence, morp_analysis, flags, p, name):
    """Some words with two syllables lose a vowel from the second syllable."""
    mapping = dictionaries.dict_rule_13

    tags = []
    for i in range(len(sentence)):
        word = normalize(sentence[i])

        if flags[i]:
            for key, value in mapping.items():

                if word.startswith(key) and flip_coin(p):
                    tags.append(f"A {i} {i+1}|||{name}|||{sentence[i]}|||REQUIRED|||-NONE-|||0")

                    word = word.replace(key, value)
                    word = capitalize(word) if is_capitalized(sentence[i]) else word
                
                    sentence[i] = word
                    flags[i] = False
                    break
        
    return tags

def rule_14(sentence, morp_analysis, flags, p, name):
    """Some words do not obey the rule_13."""
    mapping = dictionaries.dict_rule_14

    tags = []
    for i in range(len(sentence)):
        word = normalize(sentence[i])

        if flags[i]:
                
            for key, value in mapping.items():

                if word.startswith(key) and flip_coin(p):
                    tags.append(f"A {i} {i+1}|||{name}|||{sentence[i]}|||REQUIRED|||-NONE-|||0")

                    word = word.replace(key, value)
                    word = capitalize(word) if is_capitalized(sentence[i]) else word
                
                    sentence[i] = word
                    flags[i] = False
                    break
            
    return tags

def rule_15(sentence, morp_analysis, flags, p, name):
    """Last consonant of the foreign words gets transformed when they get a suffix that starts with a vowel. """
    mapping = dictionaries.dict_rule_15

    tags = []
    for i in range(len(sentence)):
        word = normalize(sentence[i])

        if flags[i]:
            for key, value in mapping.items():

                if word.startswith(key) and flip_coin(p):
                    tags.append(f"A {i} {i+1}|||{name}|||{sentence[i]}|||REQUIRED|||-NONE-|||0")

                    word = word.replace(key, value)
                    word = capitalize(word) if is_capitalized(sentence[i]) else word

                    sentence[i] = word
                    flags[i] = False
                    break
        
    return tags

def rule_16(sentence, morp_analysis, flags, p, name):
    """Some words do not obey the rule_15"""

    mapping = dictionaries.dict_rule_16

    tags = []
    for i in range(len(sentence)):
        word = normalize(sentence[i])
        
        if flags[i]:
            for key, value in mapping.items():

                if word.startswith(key) and flip_coin(p):
                    tags.append(f"A {i} {i+1}|||{name}|||{sentence[i]}|||REQUIRED|||-NONE-|||0")

                    word = word.replace(key, value)
                    word = capitalize(word) if is_capitalized(sentence[i]) else word

                    sentence[i] = word
                    flags[i] = False
                    break
        
    return tags

def rule_17(sentence, morp_analysis, flags, p, name):
    """Compound words that cointain -etmek, -edilmek, -eylemek, -olmak, -olunmak, are written seperatly unless there is a grammatic transformation."""

    mapping = dictionaries.dict_rule_17
    

    tags = []
    for i in range(len(sentence)-1):
        
        if flags[i] and flags[i+1]:

            for key, value in mapping.items():
                if (key in (normalize(sentence[i]) + " " + normalize(sentence[i+1]))) and flip_coin(p):
                    tags.append(f"A {i} {i+1}|||{name}|||{sentence[i]} {sentence[i+1]}|||REQUIRED|||-NONE-|||0")
                    sentence[i] = (normalize(sentence[i]) + " " + normalize(sentence[i+1])).replace(key, value)
                    sentence[i+1] = ""
                    flags[i] = False
                    flags[i+1] = False
                    break
    
    return tags

def rule_18(sentence, morp_analysis, flags, p, name):
    """Compound words are written jointly when they lose a vowel."""
    mapping = dictionaries.dict_rule_18 


    tags = []
    for i in range(len(sentence)):
        word = normalize(sentence[i])

        if flags[i]:

            for key, value in mapping.items():

                if word.startswith(key) and flip_coin(p):
                    tags.append(f"A {i} {i+2}|||{name}|||{sentence[i]}|||REQUIRED|||-NONE-|||0")

                    word = word.replace(key, value)
                    word = capitalize(word) if is_capitalized(sentence[i]) else word
                
                    sentence[i] = word
                    flags[i] = False
                    break
        
    return tags

def rule_19(sentence, morp_analysis, flags, p, name):
    """Compound words are written jointly when second the meaning of the second word changes."""
    mapping = dictionaries.dict_rule_19


    tags = []
    for i, word in enumerate(sentence):
        word = normalize(word)

        if flags[i]:
            for key, value in mapping.items():

                if word.startswith(key) and flip_coin(p):
                    tags.append(f"A {i} {i+2}|||{name}|||{sentence[i]}|||REQUIRED|||-NONE-|||0")

                    word = word.replace(key, value)
                    word = capitalize(word) if is_capitalized(sentence[i]) else word
                    
                    sentence[i] = word
                    flags[i] = False
                    break
        
    return tags

def rule_20(sentence, morp_analysis, flags, p, name):
    """Conjunction words created with -bilmek, -vermek, -kalmak, -durmak, -gelmek and -yazmak are written jointly if they have one of the following suffixes:
    -a, -e, -ı, -i, -u, -ü
    
    Example:
    yapıvermek, not yapı vermek
    uyuyakalmak, not uyuya kalmak
    """

    suffixes = dictionaries.dict_rule_20
    
    tags = []
    for i, word in enumerate(sentence):
        word = normalize(word)

        if flags[i]:

            for suffix in suffixes:

                if (suffix in word) and ("Verb" in morp_analysis()[i][3]) and flip_coin(p):
                    tags.append(f"A {i} {i+2}|||{name}|||{sentence[i]}|||REQUIRED|||-NONE-|||0")

                    sentence[i] = sentence[i][:word.find(suffix)+1] + ' ' + sentence[i][word.find(suffix)+1:]
                    flags[i] = False
                    break
    return tags
#  Sınava uyuyakaldığı için giremedi.
#A 1 3||uyuyakaldığı
#S Sınava uyuya kaldığı için giremedi.

def rule_21(sentence, morp_analysis, flags, p, name):
    """Conjunction words created with hane, name and zade are written jointly."""
    suffixes = dictionaries.dict_rule_21

           
    tags = []
    for i, word in enumerate(sentence):
        word = normalize(word)

        if flags[i]:
            for suffix in suffixes:

                if (suffix in word) and flip_coin(p):
                    tags.append(f"A {i} {i+2}|||{name}|||{sentence[i]}|||REQUIRED|||-NONE-|||0")

                    sentence[i] = sentence[i][:word.find(suffix)] + ' ' + sentence[i][word.find(suffix):]
                    flags[i] = False
                    break
    return tags

def rule_22(sentence, morp_analysis, flags, p, name):  
    """Conventionally, some pronounes are written jointly."""

    mapping = dictionaries.dict_rule_22

    tags = []
    for i, word in enumerate(sentence):
        word = normalize(word)

        if flags[i]:
            for key, value in mapping.items():

                if word.startswith(key) and flip_coin(p):
                    tags.append(f"A {i} {i+2}|||{name}|||{sentence[i]}|||REQUIRED|||-NONE-|||0")
                    word = word.replace(key, value)
                    word = capitalize(word) if is_capitalized(sentence[i]) else word
                                 
                    sentence[i] = word
                    flags[i] = False
                    break

    return tags

def rule_23(sentence, morp_analysis, flags, p, name):
    """First word of a sentence must be capitalized."""
    tags = []

    if flags[0] and sentence[0][0].isalpha() and is_capitalized(sentence[0]) and flip_coin(p):
        first_word = sentence[0].split(" ")[0]
        tags.append(f"A 0 1|||{name}|||{first_word}|||REQUIRED|||-NONE-|||0")
        sentence[0] = lowercase(sentence[0])
        flags[0] = False
    
    return tags

def rule_24(sentence, morp_analysis, flags, p, name):
    """In some of the words originating from Arabic and Persian, the letter a and u should be written with a circumflex accent (^)."""

    mapping = dictionaries.dict_rule_24

    tags = []
    for i, word in enumerate(sentence):
        word = lowercase(word)

        if flags[i]:

            for key, value in mapping.items():

                if word.startswith(key) and flip_coin(p):
                    tags.append(f"A {i} {i+1}|||{name}|||{sentence[i]}|||REQUIRED|||-NONE-|||0")

                    word = word.replace(key, value)
                    sentence[i] = capitalize(word) if is_capitalized(sentence[i]) else word
                    flags[i] = False
                    break
        
    return tags


def rule_25(sentence, morp_analysis, flags, p, name):
    """Grammar rules for abbreviations."""

    mapping = dictionaries.dict_rule_25

    tags = []
    for i, word in enumerate(sentence):
        
        if flags[i]:

            for key, value in mapping.items():

                if word.startswith(key) and flip_coin(p):
                    tags.append(f"A {i} {i+1}|||{name}|||{sentence[i]}|||REQUIRED|||-NONE-|||0")

                    word = word.replace(key, value)
                    sentence[i] = capitalize(word) if is_capitalized(sentence[i]) else word
                    flags[i] = False
                    break
        
    return tags