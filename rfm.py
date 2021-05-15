import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error, log_loss, roc_auc_score
#from sklearn.linear_model import LogisticRegression
from rankfm.rankfm import RankFM
import copy
import csv
from rankfm.evaluation import precision, recall

# Variables
K = 10
include_features = {
        "gender" : False,
        "age" : False,
        "occupation" : False,
        "location" : True
        }

# Read in data
def load_data(filename, path="ml-100k/"): #todo: add dataset to repo
    data = [] # user id + movie id
    y = [] # ratings
    users = set()
    items = set()
    with open(path+filename) as f:
        for line in f:
            (user, movieid, rating, ts) = line.split('\t')
            data.append({ "user_id": str(user), "item_id": str(movieid)})
            if float(rating) >= 4.0:
                    y.append(1.0)
            else:
                y.append(0.0)
            #y.append(float(rating))
            users.add(user)
            items.add(movieid)
    return (data, np.array(y), users, items)



# Load user features
def load_user_features():
    # Gets filename for user features
    filename = "user_features/feat"
    if include_features["gender"] == True:
        filename += "_g"
    if include_features["age"] == True:
        filename += "_a"
    if include_features["occupation"] == True:
        filename += "_o"
    if include_features["location"] == True:
        filename += "_l"
    filename += ".csv"

    usr_feat = pd.read_csv(filename)
    print("Loaded user features from", filename)
    return usr_feat

user_features = load_user_features()
user_features = user_features.astype(str)
print("user_features shape: ", user_features.shape)


# Create train and test sets
(train_data, y_train, train_users, train_items) = load_data("ua.base") 
(test_data, y_test, test_users, test_items) = load_data("ua.test")


X_train = pd.DataFrame(data=train_data)
X_test = pd.DataFrame(data=test_data)

unique_users = X_train.user_id.nunique()
unique_items = X_train.item_id.nunique()

# Check Matrix/Vector Dimensions
print("\n Matrix/Vector Dimensions")
print("X_train shape: {}".format(X_train.shape))
print("X_train unique users: {}".format(X_train.user_id.nunique()))
print("X_train unique items: {}".format(X_train.item_id.nunique()))


train_users = np.sort(X_train.user_id.unique())
test_users = np.sort(X_test.user_id.unique())
cold_start_users = set(test_users) - set(train_users)

train_items = np.sort(X_train.item_id.unique())
test_items = np.sort(X_test.item_id.unique())
cold_start_items = set(test_items) - set(train_items)

# User and item stats
print("\nUsers and Items")
print("X_train shape: {}".format(X_train.shape))
print("X_test shape: {}".format(X_test.shape))

print("train users: {}".format(len(train_users)))
print("test users: {}".format(len(test_users)))
print("cold-start users: {}".format(cold_start_users))

print("train items: {}".format(len(train_items)))
print("test items: {}".format(len(test_items)))
print("cold-start items: {}".format(cold_start_items))

#Evaluate X_train Matrix Sparsity
sparsity = 1 - (len(X_train) / (unique_users * unique_items))
print("\nX_train matrix sparsity: {}%".format(round(100 * sparsity, 1)))



# Build and train FM model
rankfm = RankFM(factors=20, loss='warp', max_samples=20, alpha=0.01, sigma=0.1, learning_rate=0.10, learning_schedule='invscaling')
rankfm.fit(X_train, user_features, epochs=20, verbose=True)


# Generate Model Scores for Validation Interactions
test_scores = rankfm.predict(X_test, cold_start="nan")
print("Test scores shape: ", test_scores.shape)
print(pd.Series(test_scores).describe())

# Generate TopN Recommendations for Test Users
test_recommendations = rankfm.recommend(test_users, n_items=K, filter_previous=True, cold_start="nan")
print("test_recommendations shape: ", test_recommendations.shape)

confidence_scores = test_recommendations.copy()

with open('test_recomended_items.csv','w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(["User", "Item", "Rank", "Confidence score"] )

ind = 0
for usr in range(test_recommendations.shape[0]):
    for rnk in range(test_recommendations.shape[1]):
        confidence_scores[rnk][usr] = test_scores[ind]
        with open('test_recomended_items.csv','a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow([ usr, test_recommendations[rnk][usr], rnk+1, confidence_scores[rnk][usr] ] )
        ind += 1

# Evaluate model
rankfm_precision = precision(rankfm, X_test, k=K)
rankfm_recall = recall(rankfm, X_test, k=K)

print("precision: {:.3f}".format(rankfm_precision))
print("recall: {:.3f}".format(rankfm_recall))