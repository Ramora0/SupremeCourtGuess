import os
import matplotlib.pyplot as plt
import json
import numpy as np
from fuzzywuzzy import fuzz

data_path = '/Users/leedavis/coding/Python/SupremeCourtGuess/data'

with open(f'{data_path}/basic.json', 'r') as file:
    main = json.load(file)


def justice_dict(func):
    return {
        'John G. Roberts, Jr.': func(),
        'Clarence Thomas': func(),
        'Elena Kagan': func(),
        'Ketanji Brown Jackson': func(),
        'Sonia Sotomayor': func(),
        'Samuel A. Alito, Jr.': func(),
        'Amy Coney Barrett': func(),
        'Neil Gorsuch': func(),
        'Brett M. Kavanaugh': func()
    }


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


justices = {
    'John G. Roberts, Jr.': [],
    'Clarence Thomas': [],
    'Elena Kagan': [],
    'Ketanji Brown Jackson': [],
    'Sonia Sotomayor': [],
    'Samuel A. Alito, Jr.': [],
    'Amy Coney Barrett': [],
    'Neil Gorsuch': [],
    'Brett M. Kavanaugh': []
}

for case in main:
    justice_sentiment = justice_dict(lambda: [0, 0])
    justice_count = justice_dict(lambda: [0, 0])

    # Get majority information
    majority = {}
    for vote in case['votes']:
        majority[vote['name']] = vote['vote'] == 'majority'

    Petitioner = get_petitioner(case['parties'])
    Respondent = get_respondent(case['parties'])

    majority_for_petitioner = matches_a(
        case['majority'], Petitioner, Respondent)

    # Loop through every statement
    path = f"{
        data_path}/raw_cases/{case['transcript_url'].replace('/', '-')}.json"

    with open(path, 'r') as file:
        raw_case = json.load(file)

    some_bert = False
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
            justice_for_lawyer = 0 if justice_for_petitioner == lawyer_for_petitioner else 1

            justice_sentiment[statement['speaker']][justice_for_lawyer] \
                += statement['bert_sentiment']
            justice_count[statement['speaker']][justice_for_lawyer] += 1
            some_bert = True

    if not some_bert:
        continue

    for justice in justice_sentiment:
        if justice_count[justice][0] == 0 or justice_count[justice][1] == 0:
            continue

        sentiment = justice_sentiment[justice][0] / justice_count[justice][0] - \
            justice_sentiment[justice][1] / justice_count[justice][1]

        justices[justice].append(sentiment)  # / justice_count[justice])

    # print(justices)


# Create a new figure
plt.figure(figsize=(10, 6))

# Create a boxplot for each justice
# plt.boxplot(justices.values(), labels=justices.keys(), vert=False)

for i, (justice, values) in enumerate(justices.items(), start=1):
    # Generate random jitter
    jitter = np.random.uniform(-0.3, 0.3, size=len(values))
    plt.scatter(values, np.array([i]*len(values)
                                 ) + jitter, color='black', s=25)

plt.xlabel('Sentiment')
plt.title('Justice Sentiment for Each Case')
plt.show()
