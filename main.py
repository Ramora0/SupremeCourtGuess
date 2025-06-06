from bs4 import BeautifulSoup
import json
import re

# 1: 151408
# 2: 149657
# 3: 147415
# 4: 144391
# 5: 143861
# 6: 143570

# justices = {
#     'John G. Roberts, Jr.': 'Roberts',
#     'Clarence Thomas': 'Thomas',
#     'Elena Kagan': 'Kagan',
#     'Ketanji Brown Jackson': 'Jackson',
#     'Sonia Sotomayor': 'Sotomayor',
#     'Samuel A. Alito, Jr.': 'Alito',
#     'Amy Coney Barrett': 'Barrett',
#     'Neil Gorsuch': 'Gorsuch',
#     'Brett M. Kavanaugh': 'Kavanaugh'
# }

speakers = []


def format_speaker(speaker):
    speaker = speaker.replace('Jr.', '')
    speaker = speaker.replace(',', '')
    speaker = speaker.strip()

    if speaker not in speakers:
        speakers.append(speaker)
        print(speaker)
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


file_name = '303creative'

with open("./data/"+file_name+".html", "r") as file:
    soup = BeautifulSoup(file.read(), features="html.parser")

sections = soup.find_all("section", class_='transcript-turn ng-scope')

statements = []

for section in sections:
    text = ' ' .join([p.text for p in section.find_all('p')])
    statements.append({
        'speaker': format_speaker(section.find('h4').text),
        'text': format_text(text)
    })

with open(f'./data/{file_name}.json', 'w') as file:
    file.write(json.dumps(statements))

text = '\n'.join([statement["speaker"] + ": "+statement["text"]
                 for statement in statements])

print(len(text))

with open(f'./data/{file_name}.txt', 'w') as file:
    file.write(text)
