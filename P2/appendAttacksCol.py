import pandas as pd

# Add column attack to 1 or 0
# csv_input = pd.read_csv('./Centroids/centroids10000.csv')
# csv_input['attack']=0
# # csv_input.insert(csv_input.columns, "attacks", 1, True) 
# csv_input.to_csv('./Centroids/centroids10000.csv', index=False)

# Remove first column
# df = pd.read_csv('./Centroids/centroids5000.csv')
# first_column = df.columns[0]
# df = df.drop([first_column], axis=1)
# df.to_csv('./Centroids/centroids5000.csv', index=False)

# Trim data
# df = pd.read_csv('./task3_dataset_noattacks.csv')
# df=df[::4]
# df.to_csv('./NoAttacksDownSampling/task3_dataset_noattacks_DividoEntre4.csv', index=False)

df = pd.read_csv('./IntermediateCsv/fullAttacks.csv') 
df = df.drop(['UserID','UUID','Version','TimeStemp'], axis=1)
df.to_csv('./IntermediateCsv/fullAttacksNoUUID.csv', index=False)