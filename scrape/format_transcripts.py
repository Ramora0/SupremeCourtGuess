import os
from bs4 import BeautifulSoup
import json
import re
from fuzzywuzzy import fuzz
from tqdm import tqdm


def matches_a(search, a, b):
    return fuzz.ratio(search, a) > fuzz.ratio(search, b)


def get_basic(url):
    with open('/Users/leedavis/coding/Python/SupremeCourtGuess/data/basic.json', 'r') as file:
        main = json.load(file)

    return [case for case in main if case['transcript_url'].replace('/', '-') == url][0]


speakers = []


def format_speaker(speaker):
    speaker = speaker.replace('Jr.', '')
    speaker = speaker.replace(',', '')
    speaker = speaker.strip()

    if speaker not in speakers:
        speakers.append(speaker)
        print(f'{speaker} -> {speaker.split(' ')[-1]}')
    return speaker.split(' ')[-1]


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


case_path = '/Users/leedavis/coding/Python/SupremeCourtGuess/data/'


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


def format_transcript(url):
    basic = get_basic(url)

    with open(f'{case_path}raw_cases/{url}.json', "r") as file:
        statements = json.loads(file.read())

    Petitioner = get_petitioner(basic['parties'])
    Respondent = get_respondent(basic['parties'])

    for statement in statements:
        statement["speaker"] = format_speaker(statement["speaker"])
        statement["text"] = format_text(statement["text"])

    text = '\n'.join([statement["speaker"] + ": "+statement["text"]
                      for statement in statements])
    text = f'{basic["name"]}\n{basic["facts"]}\n{basic["question"]}\n{text}'
    for petitioner in basic['petitioners']:
        supporting = petitioner['for'] \
            .replace('the Respondents', Respondent) \
            .replace('the Petitioners', Petitioner)
        text += f'\n{petitioner['name']} {petitioner['for']}'

    text += f'\nPetitioner / Appellant: {Petitioner}'
    text += f'\nRespondent / Appelee: {Respondent}'

    majority_petitioner = matches_a(
        basic['majority'], Petitioner, Respondent)

    text += '\n||\n'
    for vote in basic['votes']:
        if vote['vote'] == 'majority':
            text += (
                f"{format_speaker(vote['name'])} voted for "
                f"{'the petitioner' if majority_petitioner else 'the respondent'}\n"
            )
        else:
            text += (
                f"{format_speaker(vote['name'])} voted for "
                f"{'the respondent' if majority_petitioner else 'the petitioner'}\n"
            )

    with open(f'{case_path}formatted/{url}.txt', 'w') as file:
        file.write(text)


for file in tqdm(os.listdir(f'{case_path}raw_cases')):
    if not file.endswith('.json'):
        continue
    url = file[:-5].split('/')[-1]
    print(url)

    if not os.path.exists(f'{case_path}formatted/{url}.txt'):
        format_transcript(url)
