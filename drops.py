#!/usr/bin/env python3

import os
import json
import openai
import more_itertools
import time
import datetime
from raindropio import API, CollectionRef, Raindrop
from dataclasses import dataclass
import pickle
import hashlib
import re


now=datetime.datetime.now().strftime("%Y%m%d%H%M%S")
openai.api_key = os.environ["OPENAI_API_KEY"]
api = API(os.environ["RAINDROP_KEY"])
raindrop_collection_id=os.environ["RAINDROP_COLLECTION_ID"] # Collection ID to process

file_name = f'tags-{now}.pkl' # CHANGE ME TO LOAD EXISTING FILE

@dataclass
class Tag:
    drop_id: int
    article: str
    label: str

def load_tags(file_name):
    tags = dict()
    try: 
        with open(file_name, 'rb') as file:
            tags = pickle.load(file)
            print(f'Object successfully loaded "{file_name}", with {len(tags)} items')
    except:
        print(f'Could not load object file "{file_name}", creating new object')
    return tags

def get_drops(api, collection_id):
    page = 0
    ret = set()
    while (items := Raindrop.search(api, collection=CollectionRef({"$id": collection_id}), page=page, perpage=50)):
        print(f'Processing Page {page} from Raindrop.io')
        for item in items:
            title = re.sub('[^\w ]', '', item.title) # remove non alphanumeric characters
            title = re.sub(r'\s{2,}', ' ', title)
            ret.add((item.id,title))
        page += 1
    
    return ret


def call_gpt(items):
    if len(items) == 1:
        return [] 
    prompt = """You are a perfect classifier, you can use the following categories and only the following categories: 
    
Technology, Artificial Intelligence, NLP, Music, Programming, Product Management, Finance, Design, Coaching, User Experience, Business, Finance, Education, Social Media, Lifestyle, Entertainment, Health, Management, Other/Unknown
    
Classify each of the following internet articles into a one word category. Return Article, Label pairs, such as: 

Article: Article Name
Label: Label

---------------------------

"""
    for item in items:
        prompt += f'Article: {item}\n'

    # print('***********************\n' + prompt + '*********************\n') 

    start = time.time()
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    raw_pairs = response.choices[0].message.content
    end = time.time()
    print(f'GPT-3 took {end - start} seconds to respond')
    # print('*************************************\n GPT-3 response: ' + raw_pairs + ' \n*************************************')
    ret = dict()
    clean_response = [x for x in raw_pairs.splitlines() if not bool(re.search("^\s*$", x))]
    
    for i in range(0,len(clean_response),2):
        a = clean_response[i].replace('Article: ', '')
        l = clean_response[i+1].replace('Label: ', '')
        ret[a] = l
    
    return ret

drops = get_drops(api, raindrop_collection_id)

tags = load_tags(file_name)

for batch in more_itertools.batched(drops,15):
    to_process = []
    
    for id,item in batch:
        hash = hashlib.sha1(str(item).encode('utf-8')).hexdigest()
        if hash in tags.keys():
            print(f'Skipping {item}, already processed')
            continue
        else:
            tags[hash] = Tag(id, item, '')
            to_process.append(item)
    
    if len(to_process) <= 1:
        print(f'No new items to process, skipping batch')
        continue
    else:
        print(f'Processing {len(to_process)} items: {to_process}')

    labeled_pairs = call_gpt(to_process)    

    for article,label in labeled_pairs.items():
        hash = hashlib.sha1(str(article).encode('utf-8')).hexdigest()
        if hash in tags.keys():
            tags[hash].label = label
        else:
            print(f'Could not find article {article} in tags, GPT is hallucinating')

    with open(file_name, 'wb') as file:
        pickle.dump(tags, file)

for k,v in tags.items():
    if v.label != '':
        print(f'Updating tag "{v.article}" to: {v.label}')
        Raindrop.update(api, v.drop_id, tags=[v.label])
        time.sleep(0.5)