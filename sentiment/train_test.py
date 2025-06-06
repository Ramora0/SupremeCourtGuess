import pandas as pd
from sklearn.model_selection import train_test_split

# Load the data
data = pd.read_csv(
    '/Users/leedavis/coding/Python/SupremeCourtGuess/data/output.csv')

# Split the data into a training set and a test set
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Save the training set and the test set to CSV files
train_data.to_csv(
    '/Users/leedavis/coding/Python/SupremeCourtGuess/data/train.csv', index=False)
test_data.to_csv(
    '/Users/leedavis/coding/Python/SupremeCourtGuess/data/test.csv', index=False)
