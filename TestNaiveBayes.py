import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
import io
import string
import collections
import math
import re

# This is bad
vocabab_training = 0
len_post_type_list = []


def getVocabulary(trainData):
    # Change all words to lowercase
    newLowerTitle = trainData["Title"].str.lower()
    words = []
    tknzr = TweetTokenizer()

    # Get Tokens
    for index, row in newLowerTitle.iteritems():
        words.append(tknzr.tokenize(row))
    lemmatizer = nltk.stem.WordNetLemmatizer()

    # Remove Stop Words
    stop_words = set(stopwords.words('english'))

    vocabulary = []

    # Lemmatize  words
    for word in words:
        for w in word:
            vocabulary.append(lemmatizer.lemmatize(w))
            # Remove Punctuation and special Characters
    removePunVocabulary = [''.join(c for c in s if c not in string.punctuation) for s in vocabulary]
    removePunVocabulary = [s for s in removePunVocabulary if s]
    # newVocabulary = [re.sub('[^A-Za-z0-9]+', '', s) for s in removePunVocabulary]
    # Removing stop words
    # removePunVocabulary = [word for word in removePunVocabulary if word not in stop_words]
    removePunVocabulary.sort()
    return removePunVocabulary


def buildModel(trainData, smooth):
    print("Building Model with smoothing = ", smooth)
    # Generate Vocabulary
    print("Generate Vocabulary")
    vocabulary = getVocabulary(trainData)
    vocabulary = list(filter(None, vocabulary))
    counter = collections.Counter(vocabulary)
    vocabulary = list(counter.keys())
    # final sorted vocabulary
    # vocabulary.sort()
    with io.open('vocabulary.txt', "w", encoding="utf-8") as f:
        for item in vocabulary:
            f.write("%s\n" % item)
    # Get length of vocabulary
    vocabularyLen = len(vocabulary)
    global vocabab_training
    vocabab_training = vocabularyLen
    print("Complete Vocabulary can be found in vocabulary.txt, vocabulary length = ", vocabularyLen)

    # Get number of post types
    print("Getting number of classes")
    post_type = trainData["Post Type"]
    count_type = collections.Counter(post_type)
    types = list(count_type.keys())
    print("Classes: ", types)

    # Get vocabulary set for each post type
    for item in types:
        sdata = item + "Data"
        globals()[sdata] = trainData[trainData["Post Type"] == item]
        svocab = item + "Vocab"
        print("Getting vocabulary of class: ", item)
        globals()[svocab] = getVocabulary(globals()[sdata])
        globals()[svocab] = list(filter(None, globals()[svocab]))
        # length of vocabulary for eah post type
        count_type = collections.Counter(globals()[svocab])
        global sLen
        sLen = item + "Len"
        globals()[sLen] = len(list(count_type.keys()))
        global len_post_type_list
        print("Length of vocabulary of class: ", item, " is ", globals()[sLen])
        len_post_type_list.append(globals()[sLen])
        # frequency of each word in each post type
        print("Getting frequencies of each word in class: ", item)
        sFreq = item + "Frequency"
        globals()[sFreq] = nltk.FreqDist(globals()[svocab])

    # Generate model-2018.txt

    print("Generating model-2018.txt....")
    file = io.open("model-2018.txt", "w", encoding="utf-8")
    # with io.open('vocabulary.txt', "w", encoding="utf-8") as f:
    line = 0
    for word in vocabulary:
        s = ''
        line += 1
        print(line)
        s += str(line) + "  "
        s += word + "  "

        for item in types:
            c = globals()[item + "Frequency"].get(word)
            if type(c) != 'int':
                c = 0
            s += str(c) + "  "
            itemLen = globals()[item + "Len"]
            probability = (c + smooth) / (itemLen + vocabularyLen * smooth)
            s += format(probability, '.12f') + "  "
        s += '\r'
        file.write(s)

    print("Complete file model-2018.txt")
    file.close()


def buildDictonary(file):
    dict = {}
    model_2018 = file.readlines()
    model_2018 = [x.strip() for x in model_2018]
    print("Building  dict")

    for row in model_2018:
        row_split_list = row.split('  ')
        dict[row_split_list[1]] = row_split_list[2:]

    print("Created dict ")
    return dict


def sum_cond_prob_tokens_story(tokenList, dict, smoothing_value):
    sum = 0
    print("Cal sum of cond prob  of story")
    for token in tokenList:
        if token in dict:
            cond_prob = dict.get(token)
            prob = float(cond_prob[1])
            print("token", token, 'value is', prob)
            sum = sum + math.log10(prob)
        else:
            print("Token is not present in dict, creating cond prob of story for ", token)
            print("Gloabl vocab", vocabab_training, "len of stroy type", len_post_type_list[0])
            cond_prob = smoothing_value / (len_post_type_list[0] + vocabab_training * smoothing_value)
            sum += math.log10(cond_prob)
    return sum


def sum_cond_prob_tokens_ask_hn(tokenList, dict, smoothing_value):
    sum = 0
    print("Cal sum of cond prob of ask hn")
    for token in tokenList:
        if token in dict:
            cond_prob = dict.get(token)
            prob = float(cond_prob[3])
            print("token", token, 'value is', prob)
            sum = sum + math.log10(prob)
        else:
            print("Token is not present in dict, creating cond prob of story for ", token)
            print("Gloabl vocab", vocabab_training, "len of stroy type", len_post_type_list[0])
            cond_prob = smoothing_value / (len_post_type_list[1] + vocabab_training * smoothing_value)
            sum += math.log10(cond_prob)
    return sum


def sum_cond_prob_tokens_show_hn(tokenList, dict, smoothing_value):
    sum = 0
    print("Cal sum of cond prob of show hn")
    for token in tokenList:
        if token in dict:
            cond_prob = dict.get(token)
            prob = float(cond_prob[5])
            print("token", token, 'value is', prob)
            sum = sum + math.log10(prob)
        else:
            print("Token is not present in dict, creating cond prob of story for ", token)
            print("Gloabl vocab", vocabab_training, "len of stroy type", len_post_type_list[0])
            cond_prob = smoothing_value / (len_post_type_list[2] + vocabab_training * smoothing_value)
            sum += math.log10(cond_prob)
    return sum


def sum_cond_prob_tokens_poll(tokenList, dict, smoothing_value):
    sum = 0
    print("Cal sum of cond prob of poll")
    for token in tokenList:
        if token in dict:
            cond_prob = dict.get(token)
            prob = float(cond_prob[7])
            sum = sum + math.log10(prob)
        else:
            print("Token is not present in dict, creating cond prob of story for ", token)
            print("Gloabl vocab", vocabab_training, "len of stroy type", len_post_type_list[0])
            cond_prob = smoothing_value / (len_post_type_list[3] + vocabab_training * smoothing_value)
            sum += math.log10(cond_prob)
    return sum


def classify(tokenizeList, dict, smoothing_value, traindata):
    # prob_ask_hn = sum_cond_prob_tokens(tokenizeList, dict, "ask_hn")
    # TODO Hardcording  prob of class type have to change it
    classifed_type=''
    prob_tokens_story = 0.25 + sum_cond_prob_tokens_story(tokenizeList, dict, smoothing_value)
    prob_tokens_ask_hn = 0.25 + sum_cond_prob_tokens_ask_hn(tokenizeList, dict, smoothing_value)
    prob_tokens_show_hn = 0.25 + sum_cond_prob_tokens_show_hn(tokenizeList, dict, smoothing_value)
    prob_tokens_poll = 0.25 + sum_cond_prob_tokens_poll(tokenizeList, dict, smoothing_value)

    max_prob = max(prob_tokens_story, prob_tokens_ask_hn, prob_tokens_show_hn, prob_tokens_poll)

    # Matching where we got the max value
    if max_prob == prob_tokens_story:
        classifed_type= 'story'
    elif max_prob == prob_tokens_poll:
         classifed_type='poll'
    elif max_prob == prob_tokens_show_hn:
        classifed_type= 'show_hn'
    else:
        classifed_type= 'poll'

    return classifed_type, prob_tokens_story,prob_tokens_ask_hn,prob_tokens_show_hn,prob_tokens_poll

# Tokenize tile, Remove puncuations and lemmentize for each title
def tokenize_title(title):
    print("Tokenizing test data title")
    tweetTokenizer = TweetTokenizer()
    tokenList = tweetTokenizer.tokenize(title)
    lemmentizeList = []
    lemmatizer = nltk.stem.WordNetLemmatizer()

    for word in tokenList:
        lemmentizeList.append(lemmatizer.lemmatize(word))

    removePunTokens = [''.join(c for c in s if c not in string.punctuation) for s in lemmentizeList]
    removePunTokens = [s for s in removePunTokens if s]
    print(removePunTokens)
    return removePunTokens


def buildClassifier(testData, trainData, smoothing_value):
    # row = testData[0:1]
    # original_post_type= row["Post Type"]
    # print(row["Post Type"])
    # tokenizeList= getVocabulary(row)
    # print(tokenizeList)
    # print("Reading the contents of the model file")
    # file =open("model-2018.txt")
    # dict =buildDictonary(file)
    # classification =classify(tokenizeList,dict)
    # print("Original Post type is" , original_post_type, "Classified in" , classification)
    print("Vocab train", vocabab_training)
    print("Len of each post type", len_post_type_list)
    file = open("model-2018.txt")
    dict = buildDictonary(file)
    # testData= testData.head()
    # print("Test data is ", testData)
    print("Generating baseline-result.txt....")
    file = io.open("baseline-result.txt", "w", encoding="utf-8")
    line=0
    for index, row in testData.iterrows():
        label="right";
        s = ''
        line += 1
        s += str(line) + "  "
        s+= row["Title"]+ "  "
        tokenizeList = tokenize_title(row["Title"])
        original_post_type = row["Post Type"]
        print("Reading the contents of the model file")
        classification,prob_tokens_story,prob_tokens_ask_hn,prob_tokens_show_hn,prob_tokens_poll=classify(tokenizeList, dict, smoothing_value, trainData)
        if classification!=original_post_type:
            label="wrong"
        s+=classification + "  "
        s+= str(prob_tokens_story)+ "  "
        s+= str(prob_tokens_ask_hn)+ "  "
        s+= str(prob_tokens_show_hn)+ "  "
        s+= str(prob_tokens_poll)+ "  "
        s += str(original_post_type) + "  "
        s += label + "  "
        s += '\r'
        file.write(s)


def main():
    # Read csv File
    print("Reading csv file...")
    dataFrame = pd.read_csv("hn2018_2019.csv")
    dataFrame["Created At"] = pd.to_datetime(dataFrame["Created At"])
    dataFrame["year"] = dataFrame["Created At"].dt.year
    dataFrame["Title"] = dataFrame["Title"].str.lower()

    # Get data frames from 2018 and 2019
    print("Getting Traning Data...")
    trainData = dataFrame[dataFrame["year"] == 2018]
    print("Getting Testing Data...")
    testdata = dataFrame[dataFrame["year"] == 2019]
    buildModel(trainData, 0.5)
    smooting_value = 0.5
    buildClassifier(testdata, trainData, smooting_value)
    print("Exiting main method")

if __name__ == '__main__':
    main()
