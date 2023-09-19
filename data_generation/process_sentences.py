from morp_analyzer import Morse
from config import rule_config
from utils import fix_apostrophes_and_quotes, adjust_tag_indices, split_sentence
from tqdm import tqdm
import logging
import random
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True, help="Path to the source file")
    parser.add_argument("--experiment", type=str, default="experiment", help="Name of the experiment")
    args = parser.parse_args()

    f = open(args.source, encoding="utf-8")

    morphological_analyzer = Morse() #Morphological analyzer, implemenet yours if you want to use a different one or use a different language

    tag_file = open(f"{args.experiment}_tags.txt", "w", encoding="utf-8")

    for sentence in tqdm(f.readlines()):
        try:
            sentence = sentence.strip()
            sentence = fix_apostrophes_and_quotes(sentence) #Fix the different apostrophe and quote characters in unicode

            tags = []
            golden_sentence = split_sentence(sentence)
            source_sentence = golden_sentence.copy()

            #Lazy implementation of morpological analysis. Does not call a morphological analyzer unless it is needed.
            def get_morp_analysis():
                return morphological_analyzer.analyze(sentence.replace("\"", "'"))
            
            flags = [True] * len(source_sentence) #Flags to check if a word has not been modified.

            random.shuffle(rule_config)

            for rule_tuple in rule_config:
                rule, probability, name = rule_tuple

                current_tags = rule(source_sentence, get_morp_analysis, flags, probability, name)
                
                if current_tags:
                    tags.extend(current_tags)
            
            # Fix index errors caused by remove and insert operations
            tags = adjust_tag_indices(tags)
            # Remove empty strings in the source sentence list
            source_sentence = [x for x in source_sentence if x != ""]


            tag_file.write("S " + " ".join(source_sentence))
            tag_file.write("\n")
            tag_file.write("T " + " ".join(golden_sentence))
            tag_file.write("\n")

            for tag in tags:
                tag_file.write(tag)
                tag_file.write("\n")

            tag_file.write("\n")
                
        except Exception as e:
            print(e)
            continue

    tag_file.close()
    f.close()
