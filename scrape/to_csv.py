import os
import pandas as pd
from tqdm import tqdm

directory = '/Users/leedavis/coding/Python/SupremeCourtGuess/data/formatted'

csv_file = '/Users/leedavis/coding/Python/SupremeCourtGuess/data/output.csv'

data = {
    'cases': [],
    'responses': []
}

for filename in tqdm(os.listdir(directory)):
    with open(f'{directory}/{filename}', 'r') as file:
        lines = '\n'.join(file.readlines())
        data['cases'].append(lines.split('\n||\n')[
                             0].replace('"', "'").replace(',', ''))
        data['responses'].append(lines.split('\n||\n')[1])

df = pd.DataFrame(data)
df.to_csv(csv_file, index=False)
