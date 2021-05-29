import numpy as np
import pandas as pd

import copy
import csv
from rankfm import rankfm

from rankfm.rankfm import RankFM
from rankfm.evaluation import precision, recall

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

from uszipcode import SearchEngine

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

# Constants
K = 10

STATE = "state"
CITY = "major_city"
COUNTY="county"

LOC_TYPE = CITY

INFER_ATTR = {
        "gender" : True,
        "age" : False,
        "occupation" : False,
        "state" : False,
        "city" : False,
        "county": False
        }

INCLUDE_FEATURES = {
        "gender" : True,
        "age" : False,
        "occupation" : False,
        "state" : False,
        "city" : False,
        "county": False
        }

AGE_GROUPS = {
    0 : [0,34],
    1 : [35,45],
    2 : [46,99]
}

CLASSIFIERS = {
    "logistic_regression": True,
    "svc" : True,
    "random_forest" : True,
}






# Load interaction data
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

# load other user data -> age, gender ...
def load_user_data(filename="u.user", path="ml-100k/"):
    user_info = {}
    with open(path+filename, 'r') as fin:
        for line in fin.readlines():
            user_id, age, gender, occu, zipcode = line.split('|')
            user_info[int(user_id)] = {
                'age': int(age),
                'gender': gender,
                'occupation': occu,
                'location': str(zipcode).strip()
            }
        print('User Info Loaded!')
    return user_info

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
    elif INCLUDE_FEATURES["county"] == True:
        filename += "_t"
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



def write_recommendations_to_csv(recommendations, scores):
    confidence_scores = recommendations.copy()

    with open('test_recomended_items.csv','w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(["User", "Item", "Rank", "Confidence score"] )

    ind = 0
    for usr in range(recommendations.shape[0]):
        for rnk in range(recommendations.shape[1]):
            confidence_scores[rnk][usr] = scores[ind]
            with open('test_recomended_items.csv','a', newline='') as csvfile:
                csv_writer = csv.writer(csvfile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow([ usr, recommendations[rnk][usr], rnk+1, confidence_scores[rnk][usr] ] )
            ind += 1

def generate_recommendations(X_train, X_test, user_features, users):
    # Build and train FM model
    rankfm = RankFM( loss='warp', alpha=0.01, sigma=0.1, learning_rate=0.1, learning_schedule='invscaling')
    #rankfm = RankFM(factors=20, loss='warp', max_samples=20, alpha=0.01, sigma=0.1, learning_rate=0.10, learning_schedule='invscaling')
    rankfm.fit(X_train, user_features, epochs=20, verbose=True)

    # Generate TopN Recommendations
    recommendations = rankfm.recommend(users, n_items=K, filter_previous=True, cold_start="nan")
    print("recommendations_train shape: ", recommendations.shape)

    # Generate Model Scores for Validation Interactions
    scores = rankfm.predict(X_test, cold_start="nan")
    print("Scores shape: ", scores.shape)
    print(pd.Series(scores).describe())
    return rankfm, recommendations

def evaluate_recommender(model, X_test):
    # Evaluate model
    rankfm_precision = precision(model, X_test, k=K)
    rankfm_recall = recall(model, X_test, k=K)

    print("precision: {:.3f}".format(rankfm_precision))
    print("recall: {:.3f}".format(rankfm_recall))



def prepare_attributes_for_classifier(user_info, users, attr_type="gender"):
    attr_classes = {}
    attributes = []
    new_user_info = {}

    for i in range(len(users)):
        for key in user_info.keys():
            if int(users[i]) == key:
                new_user_info[key] = user_info[key]

    def is_in_age_group(age, age_cat):
        return True if age >= AGE_GROUPS[age_cat][0] and age <= AGE_GROUPS[age_cat][1] else False
    
    # Map zip code to state/city/county
    def map_location(zipcode, loc_type):
        search = SearchEngine(simple_zipcode=True)
        zip_code = search.by_zipcode(zipcode)
        zip_code = zip_code.to_dict()
        return zip_code[loc_type]

    for usr_id in new_user_info.keys():
        attr_value = new_user_info[usr_id][attr_type]

        if attr_type == "age":
            for age_cat in AGE_GROUPS.keys():
                if is_in_age_group(attr_value, age_cat):
                    new_attr_value = age_cat
        else:
            if attr_type == "location":
                attr_value = map_location(attr_value, LOC_TYPE)
            # Create dict of attribute labels
            if attr_value not in attr_classes.keys():
                attr_classes[attr_value] = len(attr_classes)
            new_attr_value = attr_classes[attr_value]

        # Create array of attribute representations
        attributes.append(new_attr_value)
    return attributes


# X is recommendaitons for each user (943, 10)
# z is true genders shape(943,1)
def apply_logistic_regression(X_train, X_test, y_train, y_test):
    pipe = make_pipeline(StandardScaler(), LogisticRegression())
    pipe.fit(X_train, y_train) # apply scaling on training data
    #print("True values: ", y_test)
    print("Predicted values: ", pipe.predict(X_test), "\n\n")
    print("Predicted probabilities: ", pipe.predict_proba(X_test), "\n\n")
    print("Score: ", pipe.score(X_test, y_test))
    #print("AUC: ", roc_auc_score(y_test, pipe.predict_proba(X_test), multi_class='ovr'))


def apply_svc(X_train, X_test, y_train, y_test):
    pipe = make_pipeline(StandardScaler(), svm.SVC(probability=True))
    pipe.fit(X_train, y_train) # apply scaling on training data
    #print("True values: ", y_test)
    print("Predicted values: ", pipe.predict(X_test), "\n\n")
    print("Predicted probabilities: ", pipe.predict_proba(X_test), "\n\n")
    print("Score: ", pipe.score(X_test, y_test))
    #print("AUC: ", roc_auc_score(y_test, pipe.predict_proba(X_test), multi_class='ovr'))

def apply_linear_svc(X_train, X_test, y_train, y_test):
    pipe = make_pipeline(StandardScaler(), svm.LinearSVC(probability=True))
    pipe.fit(X_train, y_train) # apply scaling on training data
    #print("True values: ", y_test)
    print("Predicted values: ", pipe.predict(X_test), "\n\n")
    print("Predicted probabilities: ", pipe.predict_proba(X_test), "\n\n")
    print("Score: ", pipe.score(X_test, y_test))
    #print("AUC: ", roc_auc_score(y_test, pipe.predict_proba(X_test), multi_class='ovr'))

def apply_nu_svc(X_train, X_test, y_train, y_test):
    pipe = make_pipeline(StandardScaler(), svm.NuSVC(probability=True))
    pipe.fit(X_train, y_train) # apply scaling on training data
    #print("True values: ", y_test)
    print("Predicted values: ", pipe.predict(X_test), "\n\n")
    print("Predicted probabilities: ", pipe.predict_proba(X_test), "\n\n")
    print("Score: ", pipe.score(X_test, y_test))
    #print("AUC: ", roc_auc_score(y_test, pipe.predict_proba(X_test), multi_class='ovr'))

def apply_random_forest(X_train, X_test, y_train, y_test):
    pipe = make_pipeline(StandardScaler(), RandomForestClassifier())
    pipe.fit(X_train, y_train) # apply scaling on training data
    #print("True values: ", y_test)
    print("Predicted values: ", pipe.predict(X_test), "\n\n")
    print("Predicted probabilities: ", pipe.predict_proba(X_test), "\n\n")
    print("Score: ", pipe.score(X_test, y_test))
    #print("AUC: ", roc_auc_score(y_test, pipe.predict_proba(X_test), multi_class='ovr'))

def classify(classifier, X_train, X_test, y_train, y_test):
    pipe = make_pipeline(StandardScaler(), classifier)
    pipe.fit(X_train, y_train)
    #print("True values: ", y_test)
    print("Predicted values: ", pipe.predict(X_test), "\n\n")
    print("Predicted probabilities: ", pipe.predict_proba(X_test), "\n\n")
    print("Score: ", pipe.score(X_test, y_test))



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

        L = int(len(user_item[usr]) * test_size)

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


    return (X_train, y_train, X_test, y_test, L)

    """
    def find_min_num_of_items(user_items):
        min_items = 1000000
        for usr in range(len(user_items)):
            if len(user_items[usr]) < min_items:
                min_items = len(user_items[usr])
        return min_items
    

    for usr in range(len(user_item)):
        if ghost_size > 0 :
            ghost_L = int(len(user_item[usr]) * ghost_size)
            user_items[usr] = user_item[usr][:-ghost_L]
        else:
            user_items[usr] = user_item[usr]

    min_items = find_min_num_of_items(user_item)
    L = int(min_items * test_size)"""


def tune_parameters(X_train, X_test, y_train,  y_test):
    # Set the parameters by cross-validation
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]},
                        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

    scores = ['precision', 'recall']

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(
            svm.SVC(), tuned_parameters, scoring='%s_macro' % score
        )
        clf.fit(X_train, y_train)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                % (mean, std * 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, clf.predict(X_test)
        print(classification_report(y_true, y_pred))
        print()


def get_classifier(clf_str):
    classifier = None
    if clf_str == "logistic_regression":
        classifier = LogisticRegression()
    elif clf_str == "svc":
        classifier = svm.SVC(probability=True)
    elif clf_str == "random_forest":
        classifier = RandomForestClassifier()
    return classifier




def get_set_users_items(x_train, x_test):
    train_users = np.sort(x_train.user_id.unique())
    test_users = np.sort(x_test.user_id.unique())
    train_items = np.sort(x_train.item_id.unique())
    test_items = np.sort(x_test.item_id.unique())
    return(train_users, train_items, test_users, test_items)


def main():
    print("Program starting... \n")

    # Load user info
    user_info = load_user_data()

    # Load user features
    user_features = load_user_features()

    # Load interaction data and create training and test sets
    (X, y, users, items, user_items, ratings) = load_data("u.data") 

    # Create train and test sets
    (X_train1, y_train1, X_test1, y_test1, L1) = prepare_splits(user_items, ratings, ghost_size=0.1)
    (X_train2, y_train2, X_test2, y_test2, L2) = prepare_splits(user_items, ratings)

    # Training and test set dimensions
    print_matrix_dim(X_train1, "X_train1")
    evaluate_matrix_sparsity(X_train1, "X_train1")

    print_matrix_dim(X_test1, "X_test1")
    evaluate_matrix_sparsity(X_test1, "X_test1")

    print_matrix_dim(X_train2, "X_train2")
    evaluate_matrix_sparsity(X_train2, "X_train2")

    print_matrix_dim(X_test2, "X_test2")
    evaluate_matrix_sparsity(X_test2, "X_test2")

    # Get train and test users
    (train_users1, train_items1, test_users1, test_items1) = get_set_users_items(X_train1, X_test1)
    (train_users2, train_items2, test_users2, test_items2) = get_set_users_items(X_train2, X_test2)

    print("L1: ", L1)
    print("L2: ", L2)

    # User and Item stats
    print("X train and test 1")
    print_user_item_stats(train_users1, test_users1, "users")
    print_user_item_stats(train_items1, test_items1, "items")
    print("X train and test 2")
    print_user_item_stats(train_users2, test_users2, "users")
    print_user_item_stats(train_items2, test_items2, "items")

    
    
    
    # Generate recommendations_train
    print("Recommender Round 1: ")
    rankfm1, recommendations_train = generate_recommendations(X_train1, X_test1, user_features, train_users1)
    evaluate_recommender(rankfm1, X_test1)

    # Generate recommendations_test
    print("Recommender Round 2: ")
    rankfm2, recommendations_test = generate_recommendations(X_train2, X_test2, user_features, train_users2)
    evaluate_recommender(rankfm2, X_test2)
    

    #Write recommendation results to file
    #write_recommendations_to_csv(recommendations_train, scores)

    

    # Classification
    attributes_train = {}
    attributes_test = {}
    classifier = None

    
    for attr in INFER_ATTR.keys():
        if INFER_ATTR[attr] == True:
            # Prepare gender attributes for classification
            attributes_train[attr] = prepare_attributes_for_classifier(user_info, test_users1, attr_type=attr)
            attributes_test[attr] = prepare_attributes_for_classifier(user_info, test_users2, attr_type=attr)
            
            print(attr, " attributes train len: ", len(attributes_train[attr]))
            print(attr, " attributes test len: ", len(attributes_test[attr]))

            # Classify attribute
            for clf in CLASSIFIERS.keys():
                if CLASSIFIERS[clf] == True:
                    print("## ", clf, " for ", attr, " ##")
                    classifier = get_classifier(clf)
                    classify(classifier, recommendations_train, recommendations_test, attributes_train[attr], attributes_test[attr])
            """print("## Logistic regression for ", attr, " ##")
            apply_logistic_regression(recommendations_train, recommendations_test, attributes_train[attr], attributes_test[attr])
            print("## Random forest for ", attr, " ##")
            apply_random_forest(recommendations_train, recommendations_test, attributes_train[attr], attributes_test[attr])
            print("### SVC for ", attr, " ##")
            apply_svc(recommendations_train, recommendations_test, attributes_train[attr], attributes_test[attr]) """
    
    """

    if INCLUDE_FEATURES["gender"] == True:
        # Prepare gender attributes for classification
        attributes_train["gender"] = prepare_attributes_for_classifier(user_info, test_users1, attr_type="gender")
        attributes_test["gender"] = prepare_attributes_for_classifier(user_info, test_users2, attr_type="gender")
        
        print("Gender attributes train len: ", len(attributes_train["gender"]))
        print("Gender attributes test len: ", len(attributes_test["gender"]))

        #tune_parameters(recommendations_train, recommendations_test, attributes_train["gender"], attributes_test["gender"])

        # Classify gender
        print("## Logistic regression ##")
        apply_logistic_regression(recommendations_train, recommendations_test, attributes_train["gender"], attributes_test["gender"])
        print("## Random forest ##")
        apply_random_forest(recommendations_train, recommendations_test, attributes_train["gender"], attributes_test["gender"])
        print("### SVC ###")
        apply_svc(recommendations_train, recommendations_test, attributes_train["gender"], attributes_test["gender"])
        #print("### LinearSVC ###")
        #apply_svc(recommendations_train, recommendations_test, attributes_train["gender"], attributes_test["gender"])
        #print("### NuSVC ###")
        #apply_svc(recommendations_train, recommendations_test, attributes_train["gender"], attributes_test["gender"])

    if INCLUDE_FEATURES["age"] == True:
        # Prepare age attributes for classification
        attributes_train["age"] = prepare_attributes_for_classifier(user_info, test_users1, attr_type="age")
        attributes_test["age"] = prepare_attributes_for_classifier(user_info, test_users2, attr_type="age")
        
        print("Age attributes train len: ", len(attributes_train["age"]))
        print("Age attributes test len: ", len(attributes_test["age"]))

        # Classify age
        print("## Logistic regression ##")
        apply_logistic_regression(recommendations_train, recommendations_test, attributes_train["age"], attributes_test["age"])
        print("## Random forest ##")
        apply_random_forest(recommendations_train, recommendations_test, attributes_train["age"], attributes_test["age"])
        print("### SVC ###")
        apply_svc(recommendations_train, recommendations_test, attributes_train["age"], attributes_test["age"])
        #print("### LinearSVC ###")
        #apply_svc(recommendations_train, recommendations_test, attributes_train["age"], attributes_test["age"])
        #print("### NuSVC ###")
        #apply_svc(recommendations_train, recommendations_test, attributes_train["age"], attributes_test["age"])

    if INCLUDE_FEATURES["occupation"] == True:
        # Prepare occupation attributes for classification
        attributes_train["occupation"] = prepare_attributes_for_classifier(user_info, test_users1, attr_type="occupation")
        attributes_test["occupation"] = prepare_attributes_for_classifier(user_info, test_users2, attr_type="occupation")
        
        print("Occupation attributes train len: ", len(attributes_train["occupation"]))
        print("Occupation attributes test len: ", len(attributes_test["occupation"]))

        # Classify occupation
        print("## Logistic regression ##")
        apply_logistic_regression(recommendations_train, recommendations_test, attributes_train["occupation"], attributes_test["occupation"])
        print("## Random forest ##")
        apply_random_forest(recommendations_train, recommendations_test, attributes_train["occupation"], attributes_test["occupation"])
        print("### SVC ###")
        apply_svc(recommendations_train, recommendations_test, attributes_train["occupation"], attributes_test["occupation"])
        #print("### LinearSVC ###")
        #apply_svc(recommendations_train, recommendations_test, attributes_train["occupation"], attributes_test["occupation"])
        #print("### NuSVC ###")
        #apply_svc(recommendations_train, recommendations_test, attributes_train["occupation"], attributes_test["occupation"])

    if INCLUDE_FEATURES["state"] == True or INCLUDE_FEATURES["city"] == True or INCLUDE_FEATURES["county"] == True:
        # Prepare location attributes for classification
        attributes_train["location"] = prepare_attributes_for_classifier(user_info, test_users1, attr_type="location")
        attributes_test["location"] = prepare_attributes_for_classifier(user_info, test_users2, attr_type="location")
        
        print("Location attributes train len: ", len(attributes_train["location"]))
        print("Location attributes test len: ", len(attributes_test["location"]))

        # Classify location
        print("## Logistic regression ##")
        apply_logistic_regression(recommendations_train, recommendations_test, attributes_train["location"], attributes_test["location"])
        print("## Random forest ##")
        apply_random_forest(recommendations_train, recommendations_test, attributes_train["location"], attributes_test["location"])
        print("### SVC ###")
        apply_svc(recommendations_train, recommendations_test, attributes_train["location"], attributes_test["location"])
        #print("### LinearSVC ###")
        #apply_svc(recommendations_train, recommendations_test, attributes_train["location"], attributes_test["location"])
        #print("### NuSVC ###")
        #apply_svc(recommendations_train, recommendations_test, attributes_train["location"], attributes_test["location"])
        

"""


"""
    # Load interaction data and create training and test sets
    #(X_train, y_train, train_users, train_items) = load_data("ua.base") 
    #(X_test, y_test, test_users, test_items) = load_data("ua.test")
    (X, y, users, items) = load_data("u.data") 
    X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, train_size=0.5, shuffle=False)
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, train_size=0.7, shuffle=False)


    #(X_train1, y_train1, train_users1, train_items1) = load_data("u1.base") 
    #(X_test1, y_test1, test_users1, test_items1) = load_data("u1.test")

    #(X_train2, y_train2, train_users2, train_items2) = load_data("u2.base") 
    #(X_test2, y_test2, test_users2, test_items2) = load_data("u2.test")
    


"""


if __name__ == "__main__":
    main()