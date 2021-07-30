import pandas as pd

"""
recommendations_train = pd.read_csv('data/bx-pre/rec_train.csv', sep=',')
recommendations_test = pd.read_csv('data/bx-pre/rec_test.csv', sep=',')

hits = 0
for ind in recommendations_train.index:
    for i in range(50):
        for j in range(50):
            if recommendations_train[str(i)][ind] == recommendations_test[str(j)][ind]:
                hits += 1

hr = hits/943

print("hits: ", hits)
print("hit_rate: ", hr)

"""

state_dist = {'CA': 116, 'MN': 78, 'NY': 60, 'TX': 51, 'IL': 50, 'None': 37, 'MA': 35, 'PA': 34, 'OH': 32, 'MD': 27, 'VA': 27, 'FL': 24, 'WA': 24, 'MI': 23, 'WI': 22, 'OR': 20, 'CO': 20, 'GA': 19, 'NC': 19, 'NJ': 18, 'CT': 17, 'MO': 17, 'AZ': 14, 'IA': 14, 'DC': 14, 'TN': 12, 'SC': 11, 'KY': 11, 'IN': 9, 'UT': 9, 'OK': 9, 'ID': 7, 'NH': 6, 'NE': 6, 'LA': 6, 'VT': 5, 'AK': 5, 'KS': 4, 'NV': 3, 'AL': 3, 'MS': 3, 'RI': 3, 'WV': 3, 'DE': 3, 'ND': 2, 'MT': 2, 'NM': 2, 'HI': 2, 'ME': 2, 'AR': 1, 'WY': 1, 'SD': 1}
for key in state_dist.keys():
    state_dist[key] = state_dist[key] / 943 * 100


print(state_dist)



