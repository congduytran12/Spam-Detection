import pandas as pd

# read csv file
df = pd.read_csv('enron_spam_data.csv')

# choose the desired columns
df_filtered = df[['Spam/Ham', 'Message']]

# rename column headers
df_filtered.rename(columns={'Spam/Ham': 'label', 'Message': 'text'}, inplace=True)

# drop rows with empty text values
df_filtered.dropna(subset=['text'], inplace=True)

# convert cells to single line
df_filtered['text'] = df_filtered['text'].apply(lambda x: x.replace('\n', ' ') if isinstance(x, str) else x)

# save the modified data to new csv file
df_filtered.to_csv('enron_spam.csv', index=False)
