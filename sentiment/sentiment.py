import json
import os
from fuzzywuzzy import fuzz
from tqdm import tqdm
import re
import spacy
from transformers import pipeline, AutoModel
import torch

# print('loading nlp')
# nlp = spacy.load("en_core_web_sm")

# print('loading sentiment analysis')
# sentiment_analysis = pipeline(
#     "sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

model = AutoModel.from_pretrained('nlpaueb/legal-bert-small-uncased')

model.load_state_dict(torch.load(
    '/Users/leedavis/coding/Python/SupremeCourtGuess/data/court_guess_model.pth', map_location=torch.device('cpu')))

sentiment_analysis = pipeline("sentiment-analysis", model=model)


def get_sentiment(text):
    text = format_text(text)

    if (len(text.split(' ')) < 5):
        return 0

    try:
        result = sentiment_analysis(text)
    except:
        return 0

    sentiment_score = result[0]['score']

    return sentiment_score * (1 if result[0]['label'] == 'POSITIVE' else -1)


def sub_all(pattern, sub, text):
    new_text = re.sub(pattern, sub, text, flags=re.IGNORECASE)
    while new_text != text:
        text = new_text
        new_text = re.sub(pattern, sub, text, flags=re.IGNORECASE)

    return text


def format_text(text):
    text = text.strip()
    text = sub_all(r'\n', '', text)
    text = sub_all(r'  ', ' ', text)
    text = sub_all(r'(\w+)\s--\s\1', r'\1', text)
    text = sub_all(r'(\w+)\s(\w+)\s--\s\1\s\2', r'\1 \2', text)

    return text


def petitioner(position):
    if 'Petitioner' in position or 'Appellant' in position:
        return 1
    elif 'Respondent' in position or 'Appellee' in position:
        return -1
    # print(f'{position} is not a party')
    return 0


def matches_a(search, a, b):
    return fuzz.ratio(search, a) > fuzz.ratio(search, b)


def get_petitioner(parties):
    if 'Petitioner' in parties:
        return parties['Petitioner']
    elif 'Appellant' in parties:
        return parties['Appellant']


def get_respondent(parties):
    if 'Respondent' in parties:
        return parties['Respondent']
    elif 'Appellee' in parties:
        return parties['Appellee']


def get_justice_sentiment(case):
    path = f"{data_path}/raw_cases/{case['transcript_url'].replace('/', '-')}" + \
        '.json'
    # print(path)
    if not os.path.exists(path):
        print("no path :(")
        return 0

    with open(path, 'r') as file:
        transcript = json.load(file)

    # Petitioner = get_petitioner(case['parties'])
    # Respondent = get_respondent(case['parties'])

    # majority_for_petitioner = matches_a(
    #     case['majority'], Petitioner, Respondent)

    # net_sentiment = {}
    # for justice in justices:
    #     net_sentiment[justice] = 0

    for i in (range(len(transcript)-1)):
        statement = transcript[i]
        next_statement = transcript[i+1]

        if "bert_sentiment" in statement:
            continue

        if statement['speaker'] not in justices:
            continue
        lawyer = [lawyer for lawyer in case['petitioners']
                  if lawyer['name'] == next_statement['speaker']]
        if len(lawyer) == 0:
            continue
        lawyer = lawyer[0]

        # majority = [justice for justice in case['votes']
        #             if justice['name'] == statement['speaker']][0]["vote"] == 'majority'

        # justice_for_petitioner = 1 if majority == majority_for_petitioner else -1

        # lawyer_for_petitioner = petitioner(lawyer['for'])

        # print()
        # print(str(statement)+' | ' + str(next_statement))

        # Dw this formats the text
        sentiment = get_sentiment(statement['text'])
        transcript[i]['custom_bert_sentiment'] = sentiment

        # net_sentiment[statement['speaker']] += sentiment * \
        #     justice_for_petitioner * lawyer_for_petitioner

    with open(path, 'w') as file:
        file.write(json.dumps(transcript))

    # return net_sentiment


justices = [
    'John G. Roberts, Jr.', 'Clarence Thomas', 'Elena Kagan', 'Ketanji Brown Jackson', 'Sonia Sotomayor', 'Samuel A. Alito, Jr.', 'Amy Coney Barrett', 'Neil Gorsuch', 'Brett M. Kavanaugh'
]

data_path = '/Users/leedavis/coding/Python/SupremeCourtGuess/data'

with open(f'{data_path}/basic.json', 'r') as file:
    main = json.load(file)

# get_justice_sentiment(main[0])

for case in tqdm(main):
    get_justice_sentiment(case)
