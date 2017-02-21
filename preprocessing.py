import datetime
import csv
import os
import json

from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from os import listdir

def getTime():
    return str(datetime.datetime.time(datetime.datetime.now()))

def tlog(msg):
    print("["+getTime()+"] "+msg)

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def review_to_JSON(review):
    json_review = {}
    json_review['text'] = review[1]
    json_review['pos_score'] = review[2]
    json_review['neg_score'] = review[3]
    json_string = json.dumps(json_review)
    return json_string

def getSentimentDictionary(filename):
    sentiment_dict = dict()
    with open(filename) as file:
        for row in file:
            if row.startswith("a") or row.startswith("v"):
                temp_list = row.split()[0:4]
                word = row.split()[4].split("#")[0]
                if word not in sentiment_dict:
                    sentiment_dict[word] = [float(row.split()[2]), float(row.split()[3])]
    return sentiment_dict

def process_set(paths, sent_dict):
    neg_but_pos = 0
    pos_but_neg = 0
    reviews = []
    for path in paths:
        fileset = [f for f in listdir(path)]
        for f in fileset:
            with open(path+"/"+f, 'r', encoding="utf8") as content_file:
                content = content_file.read()
            soup = BeautifulSoup(content.lower(), 'html.parser')
            tokens = word_tokenize(soup.getText())
            tagged_tokens = pos_tag(tokens)

            review_str = ""
            not_flag = False
            word_count = 0
            pos_score = 0
            neg_score = 0
            for (x,y) in tagged_tokens:
                if y.startswith("JJ") or y.startswith("RB") or y.startswith("NN") or y.startswith("V"):
                    review_str += x + " "

                    scores = sent_dict.get(x, [0,0])
                    pos = scores[0]
                    neg = scores[1]

                    if not_flag:
                        pos_score += neg
                        neg_score += pos
                    else:
                        pos_score += pos
                        neg_score += neg

                elif x == "no" or x == "not":
                    not_flag = True
                else:
                    not_flag = False

            if pos_score > neg_score:
                if "neg" in path:
                    pos_but_neg += 1
            elif neg_score > pos_score:
                if "pos" in path:
                    neg_but_pos += 1
            reviews.append(("processed_"+path+"/"+f.split('.')[0]+".json", review_str, pos_score, neg_score))

    tlog("Positive reviews with negative score: " + str(neg_but_pos))
    tlog("Negative reviews with positive score: " + str(pos_but_neg))
    return reviews

tlog("Reading sentiment dictionary.")
sent_dict = getSentimentDictionary("SentiWordNet.txt")
tlog("Sentiment dictionary read.")

tlog("Reading reviews and removing html elements.")
reviews = process_set(("data/train/pos", "data/train/neg", "data/test"), sent_dict)
tlog("Reviews processed.")

tlog("Writing processed files to disk.")

#Create processed directories if they don't exist
make_dir("processed_data")
make_dir("processed_data/train")
make_dir("processed_data/test")
make_dir("processed_data/train/pos")
make_dir("processed_data/train/neg")

for review in reviews:
    f = open(review[0], 'w', encoding="utf8")
    f.write(review_to_JSON(review))
    f.close()

tlog("Done.")
