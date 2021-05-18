import numpy as np
import pandas as pd

from rankfm.rankfm import RankFM
import copy
import csv
from rankfm.evaluation import precision, recall

# Constants
K = 10

INCLUDE_FEATURES = {
        "gender" : False,
        "age" : False,
        "occupation" : False,
        "state" : True,
        "city" : False
        }


# Load interaction data
def load_data(filename, path="ml-100k/"):
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
            users.add(user)
            items.add(movieid)
    return (data, np.array(y), users, items)



# Load user features
def load_user_features():
    # Gets filename for user features
    filename = "user_features/feat"
    if INCLUDE_FEATURES["gender"] == True:
        filename += "_g"
    if INCLUDE_FEATURES["age"] == True:
        filename += "_a"
    if INCLUDE_FEATURES["occupation"] == True:
        filename += "_o"
    if INCLUDE_FEATURES["state"] == True:
        filename += "_s"
    elif INCLUDE_FEATURES["city"] == True:
        filename += "_c"
    filename += ".csv"

    usr_feat = pd.read_csv(filename)
    print("Loaded user features from", filename)
    usr_feat = usr_feat.astype(str)
    print("User features shape: ", usr_feat.shape, "\n")
    return usr_feat

# Print Matrix Dimensions for training/test sets
def print_matrix_dim(x_set, set_name=""):
    print("Matrix Dimensions for ", set_name)
    print(set_name, " shape: {}".format(x_set.shape))
    print(set_name, " unique users: {}".format(x_set.user_id.nunique()))
    print(set_name, " unique items: {}".format(x_set.item_id.nunique()))
    print("\n")


# Get cold-start users and items
def get_coldstart_units(train_units, test_units, unit_name="units"):
    cold_start_units = set(test_units) - set(train_units)
    return cold_start_units

def prepare_set(t_data, y_train, train_users, train_items, train=True):
    x_set = pd.DataFrame(data=t_data)
    t_users = np.sort(x_set.user_id.unique())
    t_items = np.sort(x_set.item_id.unique())

    return x_set, t_users, t_items

# Prints user and item stats
def print_user_item_stats(train_units, test_units, unit_name="units"):
    print("Stats for ", unit_name)
    print("Train ", unit_name, ": {}".format(len(train_units)))
    print("Test ", unit_name, "{}".format(len(test_units)))
    cold_start_units = get_coldstart_units(train_units, test_units, unit_name)
    print("cold-start ", unit_name, ": {}".format(cold_start_units))
    print("\n")


#Evaluate X_train Matrix Sparsity
def evaluate_matrix_sparsity(x_set, set_name=""):
    unique_users = x_set.user_id.nunique()
    unique_items = x_set.item_id.nunique()
    sparsity = 1 - (len(x_set) / (unique_users * unique_items))
    print(set_name, " matrix sparsity: {}%".format(round(100 * sparsity, 1)))
    print("\n")




"""

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

"""

def main():
    print("Program starting... \n")

    # Load user features
    user_features = load_user_features()

    #TODO: train/test users/items are defined twice, decide which method to use
    #TODO: maybe just move the preparations into the load data?

    # Load interaction data and create training and test sets
    (train_data, y_train, train_users, train_items) = load_data("ua.base") 
    (test_data, y_test, test_users, test_items) = load_data("ua.test")

    # Prepare training sets
    X_train, train_users, train_items = prepare_set(train_data, y_train, train_users, train_items)
    X_test, test_users, test_items = prepare_set(test_data, y_test, test_users, test_items)

    

    # Training and test set dimensions
    print_matrix_dim(X_train, "X_train")
    evaluate_matrix_sparsity(X_train, "X_train")

    print_matrix_dim(X_test, "X_test")

    # User and Item stats
    print_user_item_stats(train_users, test_users, "users")
    print_user_item_stats(train_items, test_items, "items")

    



if __name__ == "__main__":
    main()