import json
import os


data_path = '/Users/leedavis/coding/Python/SupremeCourtGuess/data'


def get_path(case):
    return f"{data_path}/raw_cases/{case['transcript_url'].replace('/', '-')}.json"


# Load the JSON file
with open('/Users/leedavis/coding/Python/SupremeCourtGuess/data/basic.json', 'r') as file:
    data = json.load(file)

print(len(data))
filtered_data = [case for case in data if len(case['votes']) <= 9]
print(len(filtered_data))


filtered_data = [case for case in data if os.path.exists(get_path(case))]
print(len(filtered_data))

# Write the filtered list back to the JSON file
with open('/Users/leedavis/coding/Python/SupremeCourtGuess/data/new_basic.json', 'w') as file:
    json.dump(filtered_data, file)
