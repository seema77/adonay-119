import nltk  #natural lang toolkit
from nltk.stem import PorterStemmer
stemmer=PorterStemmer()

import json
import pickle # 0 1
import numpy as np 

words=[] # input coming from user
classes=[] #categories greeting,goodbye
word_tags_list=[]
ignore_words=['?','!',',','.',"'s","'m"] # I'm
train_data_file=open('intents.json').read()
intents=json.loads(train_data_file)

def get_stem_words(words,ignore_words):
    stem_words=[]
    for word in words:  #i am not feeling good
        if word not in ignore_words:
            w=stemmer.stem(word.lower())
            stem_words.append(w)
    return stem_words

for intent in intents['intents']:
    for pattern in intent['patterns']:
        pattern_word=nltk.word_tokenize(pattern) # how are you= "how" "are" "you"
        words.extend(pattern_word)
        word_tags_list.append((pattern_word,intent['tag']))
    if intent['tag'] not in classes:
        classes.append(intent['tag'])
        stem_words=get_stem_words(words,ignore_words)

#print(stem_words)
print("------------------------------------>")
#print(word_tags_list)
#print(classes)

def create_bot_corpus(stem_words,classes):
    stem_words=sorted(list(set(stem_words)))
    classes=sorted(list(set(classes)))

    pickle.dump(stem_words,open('words.pkl','wb'))
    pickle.dump(classes,open('classes.pkl','wb'))

    return stem_words,classes
stem_words,classes=create_bot_corpus(stem_words,classes)
print(stem_words)
print("------------------------------------>")
print(word_tags_list[0])

for word_tags in word_tags_list:
    bag_of_words=[]
    pattern_words=word_tags[0]
    for word in pattern_words:
        index=pattern_words.index(word) #["hi=0 there=1"]
        word=stemmer.stem(word.lower())
        pattern_words[index]=word
    for word in stem_words:
        if word in pattern_words:
            bag_of_words.append(1)
        else:
            bag_of_words.append(0)
    print(bag_of_words)