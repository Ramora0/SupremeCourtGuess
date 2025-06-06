import json
from fuzzywuzzy import fuzz
import re
from tqdm import tqdm

data_path = '/Users/leedavis/coding/Python/SupremeCourtGuess/data'


def matches_a(search, a, b):
    return fuzz.ratio(search, a) > fuzz.ratio(search, b)


def get_petitioner(parties):
    if 'Petitioner' in parties:
        return parties['Petitioner']
    elif 'Appellant' in parties:
        return parties['Appellant']
    return None


def get_respondent(parties):
    if 'Respondent' in parties:
        return parties['Respondent']
    elif 'Appellee' in parties:
        return parties['Appellee']
    return None


def petitioner(position):
    if 'Petitioner' in position or 'Appellant' in position:
        return 1
    elif 'Respondent' in position or 'Appellee' in position:
        return -1
    return 0


def sub_all(pattern, sub, text):
    new_text = re.sub(pattern, sub, text, flags=re.IGNORECASE)
    while new_text != text:
        text = new_text
        new_text = re.sub(pattern, sub, text, flags=re.IGNORECASE)

    return text


def format_text(text):
    text = text.strip()
    text = sub_all(r'"', "'", text)
    text = sub_all(r'\n', '', text)
    text = sub_all(r'  ', ' ', text)
    text = sub_all(r'(.+)\s--\s\1', r'\1', text)
    # text = sub_all(r'(\w+)\s(\w+)\s--\s\1\s\2', r'\1 \2', text)

    return text


justices = [
    'John G. Roberts, Jr.',
    'Clarence Thomas',
    'Elena Kagan',
    'Ketanji Brown Jackson',
    'Sonia Sotomayor',
    'Samuel A. Alito, Jr.',
    'Amy Coney Barrett',
    'Neil Gorsuch',
    'Brett M. Kavanaugh'
]


def add_ideal_sentiment(case, output_csv):
    path = f"{data_path}/raw_cases/{case['transcript_url'].replace('/', '-')}" + \
        ".json"

    majority = {}
    for vote in case['votes']:
        majority[vote['name']] = vote['vote'] == 'majority'

    Petitioner = get_petitioner(case['parties'])
    Respondent = get_respondent(case['parties'])

    majority_for_petitioner = matches_a(
        case['majority'], Petitioner, Respondent)

    with open(path, 'r') as file:
        raw_case = json.load(file)

    for i, statement in enumerate(raw_case[1:-1]):
        if statement['speaker'] in justices and 'bert_sentiment' in statement:
            lawyer = [lawyer for lawyer in case['petitioners']
                      if lawyer['name'] == raw_case[i+2]['speaker']]
            if len(lawyer) == 0:
                print('cant find lawyer ' + raw_case[i+2]['speaker'])
                continue
            lawyer = lawyer[0]

            lawyer_for_petitioner = petitioner(lawyer['for'])
            justice_for_petitioner = 1 if majority == majority_for_petitioner else -1

            text = format_text(statement['text'])

            if len(text.split(' ')) < 10 or lawyer_for_petitioner == 0:
                continue

            output_csv.write(
                f"\"{text}\",{lawyer_for_petitioner * justice_for_petitioner}\n")


with open(f'{data_path}/output.csv', 'w') as output_csv:
    output_csv.write('question,sentiment\n')

    with open(f'{data_path}/basic.json', 'r') as file:
        main = json.load(file)

    for case in tqdm(main):
        add_ideal_sentiment(case, output_csv)
