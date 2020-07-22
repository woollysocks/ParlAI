#!/usr/bin/env python3

"""
Pre-processing script to segment the context paragraphs from SQuAD2 into single sentences. 
Creates new examples per sentence.
"""

import os
import json
import spacy
import copy

nlp = spacy.load("en_core_web_sm")

path = "~/Documents/General/nlp_research/data/SQuAD2/"
train_datapath = path + "train-v2.0.json"
dev_datapath = path + "dev-v2.0.json"
new_train_file = open(os.path.join(path, "segmented", "train-seg-v2.0.json"), "w")
new_dev_file = open(os.path.join(path, "segmented", "dev-seg-v2.0.json"), "w")

def segment_data(path):
    passed = 0
    kept = 0
    with open(path) as data_file:
        squad = json.load(data_file)
    squad_sentenced = copy.deepcopy(squad['data'])
    for i, article in enumerate(squad['data']):
        # each paragraph is a context for the attached questions
        for j, paragraph in enumerate(article['paragraphs']):      
            context = paragraph['context']
            context_sentences = list(nlp(context).sents)
            for sentence in context_sentences:
                # convert from spacy span back to string
                sentence = str(sentence)
                # jankily handle bad sentence segmentation and short sentences
                if (sentence[-1] not in ['.', '!', '?', '"', '\'', '`'] and sentence[-2:] not in ['.)', '.]']) or (sentence[0] != sentence[0].upper()):
                    passed += 1
                    pass
                else:
                    kept += 1
                    new_paragraph = {}
                    new_paragraph['qas'] = paragraph['qas']
                    new_paragraph['context'] = sentence
                    squad_sentenced[i]['paragraphs'].append(new_paragraph)
            squad_sentenced[i]['paragraphs'].remove(paragraph)
    print(passed, kept)
    new_squad = {}
    new_squad['version'] = squad['version']
    new_squad['data'] = squad_sentenced
    return new_squad

new_dev = segment_data(dev_datapath)
json.dump(new_dev, new_dev_file)
print("Done with dev data!")

new_train = segment_data(train_datapath)
json.dump(new_train, new_train_file)
print("Done with training data!")
