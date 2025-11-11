import pandas as pd

# Load the dataframes
train_df = pd.read_csv('data/UNSW_NB15_training-set.csv')
test_df = pd.read_csv('data/UNSW_NB15_testing-set.csv')

print(f"Training set size: {train_df.shape}")
print(f"Testing set size: {test_df.shape}")