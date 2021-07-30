import pandas as pd

users = [{'user_id': '1', 'timestamp': 100}, {'user_id': '2', 'timestamp': 101},
        {'user_id': '1', 'timestamp': 99}, {'user_id': '2', 'timestamp': 30},
        {'user_id': '1', 'timestamp': 50}, {'user_id': '2', 'timestamp': 1088},
        {'user_id': '1', 'timestamp': 1709}, {'user_id': '2', 'timestamp': 6},
        {'user_id': '1', 'timestamp': 11111}]

data = pd.DataFrame(data=users)
"""

data.sample(frac=1)
train_size = 0.5
ranks = data.groupby('user_id')['timestamp'].rank(method='first')
print(ranks)
counts = data['user_id'].map(data.groupby('user_id')['timestamp'].apply(len))
print(counts)
thrs_train = (ranks / counts) <= train_size
thres_train = pd.DataFrame(thrs_train, columns=["thrs_train"])
data = data.join(thres_train)
data=data.sort_values('user_id')
print("data: ", data)
X_train = data[data['thrs_train'] == True]
print("X_train: ", X_train)



"""
train_size = 0.5

data = data.sample(frac=1).reset_index(drop=True)
print(data)

data['ranks'] = data.groupby('user_id').cumcount()
data['ranks'] = data['ranks'].transform(lambda x: x+1)
print(data['ranks'])
data['counts'] = data.groupby(["user_id"])["user_id"].transform("count")
print(data['counts'])
data['thrs_train'] = (data['ranks'] / data['counts']) <= train_size
#thres_train = pd.DataFrame(thrs_train, columns=["thrs_train"])
#data = data.join(thres_train)


data=data.sort_values('user_id')
print("data: ", data)