import numpy as np
import pandas as pd

import copy

def load_data(filename, path="ml-100k/"):
    data = [] # user id + movie id
    y = [] # ratings
    users = []
    items = []
    data_user_item_tmp = {}
    data_user_item = {}
    ratings = np.zeros((943, 1682), dtype=np.double)
    with open(path+filename) as f:
        for line in f:
            (user, movieid, rating, ts) = line.split('\t')
            data.append({ "user_id": str(user), "item_id": str(movieid)})
            #data_inds[int(user)-1][int(movieid)-1] = len(data)
            if float(rating) >= 4.0:
                    y.append(1.0)
                    ratings[int(user) - 1, int(movieid) - 1] = 1.0
            else:
                y.append(0.0)
                ratings[int(user) - 1, int(movieid) - 1] = 0.0
            users.append(user)
            items.append(movieid)

            # build user-item dict (based on timestamp)
            user = int(user)
            movieid = int(movieid)
            if user - 1 not in data_user_item_tmp:
                data_user_item_tmp[user - 1] = [ [movieid- 1, ts] ]
            else:
                data_user_item_tmp[user - 1].append([ movieid - 1, ts ])

    # sort each users' items
    for key, items in data_user_item_tmp.items():
        sorted_items = sorted(items, key=lambda x:x[1]) # sorted on timestamp
        data_user_item[key] = [item[0] for item in sorted_items] #items are sorted on timestamp

    # Prepare data
    data = pd.DataFrame(data=data)
    y = np.array(y)
    users = np.array(users)
    users = np.sort(np.unique(users))
    items = np.array(items)
    items = np.sort(np.unique(items))
    return (data, y, users, items, data_user_item, ratings)


def prepare_splits( user_item, ratings, test_size=0.1, ghost_size=0):
    user_items = {}
    user_items_train = {}
    user_items_test = {}
    X_train = []
    y_train = []
    X_test = []
    y_test = []

    for usr in range(len(user_item)):
        if ghost_size > 0 :
            ghost_L = int(len(user_item[usr]) * ghost_size)
            user_items[usr] = user_item[usr][:-ghost_L]
        else:
            user_items[usr] = user_item[usr]

        L = int(len(user_items[usr]) * test_size)
        user_items_test[usr] = user_items[usr][-L:]
        user_items_train[usr] = user_items[usr][:-L]

        for item in user_items_train[usr]:
            X_train.append({ "user_id": str(usr+1), "item_id": str(item+1)})
            y_train.append(ratings[usr, item])

        for item in user_items_test[usr]:
            X_test.append({ "user_id": str(usr+1), "item_id": str(item+1)})
            y_test.append(ratings[usr, item])
    
    X_train = pd.DataFrame(data=X_train)
    y_train = np.array(y_train)
    X_test = pd.DataFrame(data=X_test)
    y_test = np.array(y_test)


    return (X_train, y_train, X_test, y_test)





# Get cold-start users and items
def get_coldstart_units(train_units, test_units, unit_name="units"):
    cold_start_units = set(test_units) - set(train_units)
    return cold_start_units


# Prints user and item stats
def print_user_item_stats(train_units, test_units, unit_name="units"):
    print("Stats for ", unit_name)
    print("Train ", unit_name, ": {}".format(len(train_units)))
    print("Test ", unit_name, "{}".format(len(test_units)))
    cold_start_units = get_coldstart_units(train_units, test_units, unit_name)
    print("cold-start ", unit_name, ": {}".format(cold_start_units))
    print("\n")

# Print Matrix Dimensions for training/test sets
def print_matrix_dim(x_set, set_name=""):
    print("Matrix Dimensions for ", set_name)
    print(set_name, " shape: {}".format(x_set.shape))
    print(set_name, " unique users: {}".format(x_set.user_id.nunique()))
    print(set_name, " unique items: {}".format(x_set.item_id.nunique()))
    print("\n")


#Evaluate X_train Matrix Sparsity
def evaluate_matrix_sparsity(x_set, set_name=""):
    unique_users = x_set.user_id.nunique()
    unique_items = x_set.item_id.nunique()
    sparsity = 1 - (len(x_set) / (unique_users * unique_items))
    print(set_name, " matrix sparsity: {}%".format(round(100 * sparsity, 1)))
    print("\n")

def main():
    (X, y, users, items, user_items, ratings) = load_data("u.data") 
    print("loading data complete")
    (X_train1, y_train1, X_test1, y_test1) = prepare_splits(user_items, ratings, ghost_size=0.1)
    (X_train2, y_train2, X_test2, y_test2) = prepare_splits(user_items, ratings)
    
    
    print("X_train1 shape: ", X_train1.shape)
    print("X_test1 shape: ", X_test1.shape)
    print("y_train1 shape: ", y_train1.shape)
    print("y_test1 shape: ", y_test1.shape)

    print("X_train2 shape: ", X_train2.shape)
    print("X_test2 shape: ", X_test2.shape)
    print("y_train2 shape: ", y_train2.shape)
    print("y_test2 shape: ", y_test2.shape)

    unique_users = X.user_id.nunique()
    unique_items = X.item_id.nunique()
    unique_users_train1 = X_train1.user_id.nunique()
    unique_items_train1 = X_train1.item_id.nunique()
    unique_users_test1 = X_test1.user_id.nunique()
    unique_items_test1 = X_test1.item_id.nunique()

    unique_users_train2 = X_train2.user_id.nunique()
    unique_items_train2 = X_train2.item_id.nunique()
    unique_users_test2 = X_test2.user_id.nunique()
    unique_items_test2 = X_test2.item_id.nunique()

    print("X unique users: ", unique_users)
    print("X unique items: ", unique_items)
    print("X_train1 unique users: ", unique_users_train1)
    print("X_train1 unique items: ", unique_items_train1)
    print("X_test1 unique users: ", unique_users_test1)
    print("X_test1 unique items: ", unique_items_test1)
    print("X_train2 unique users: ", unique_users_train2)
    print("X_train2 unique items: ", unique_items_train2)
    print("X_test2 unique users: ", unique_users_test2)
    print("X_test2 unique items: ", unique_items_test2)

    evaluate_matrix_sparsity(X, "X")

    evaluate_matrix_sparsity(X_train1, "X_train1")

    evaluate_matrix_sparsity(X_test1, "X_test1")

    evaluate_matrix_sparsity(X_train2, "X_train2")

    evaluate_matrix_sparsity(X_test2, "X_test2")


    


if __name__ == "__main__":
    main()