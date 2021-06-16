# Imports
import numpy as np
import pandas as pd
from uszipcode import SearchEngine


# Utility imports
import copy
import csv

# RankFM imports
from rankfm import rankfm
from rankfm.rankfm import RankFM
from rankfm.evaluation import precision, recall, hit_rate

# Sklearn classifier imports
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.dummy import DummyClassifier

# Sklearn metric imports
from sklearn.metrics import roc_curve, auc

# Sklearn pipeline imports
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

# Sklearn hyperparameter optimization imports
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report




# Constants
N = 50

STATE = "state"
CITY = "major_city"
COUNTY="county"

LOC_TYPE = STATE

INFER_ATTR = {
        "gender" : True,
        "age" :  True,
        "occupation" : False,
        "location": False
        }

INCLUDE_FEATURES = {
        "gender" : True,
        "age" : True,
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
    "dummy" : True,
    "log_reg": True,
    "svc" : True,
    "ran_for" : True
}

NUM_USERS = 943
NUM_ITEMS = 1682

statistics = {
    "males" : 0,
    "age0": 0,
    "age1": 0,
    "age2": 0,

}

### Data Loader functions ###

# Load interaction data
def load_interaction_data(filename="u.data", path="ml-100k/"):
    data = [] # user id + movie id
    ratings = [] #TODO
    with open(path+filename) as f:
        for line in f:
            (user, movieid, rating, ts) = line.split('\t')
            data.append({ "user_id": int(user), "item_id": int(movieid), "ts": int(ts.strip())})


    # Prepare data
    data = pd.DataFrame(data=data)
    #print(data.iloc[0])
    return data


# load user data
def load_user_data(filename="u.user", path="ml-100k/"):
    user_info = {}
    #user_features = []
    with open(path+filename, 'r') as fin:
        for line in fin.readlines():
            user_id, age, gender, occu, zipcode = line.split('|')
            user_info[int(user_id)] = {
                'age': int(age),
                'gender': gender,
                'occupation': occu,
                'location': str(zipcode).strip()
            }
            if gender == "M":
                statistics["males"] += 1
            if int(age) < 35:
                statistics["age0"] += 1 
            elif  int(age) <= 45:
                statistics["age1"] += 1   
            elif int(age) < 99:
                statistics["age2"] += 1   
            #user_features.append({"user_id": str(user_id)})  
        print('User Info Loaded!')
    return user_info
    #return (user_info, user_features)

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


### Data preparation functions ###

def prepare_splits(data, train_size=0.9, test_size=0.1):
    ranks = data.groupby('user_id')['ts'].rank(method='first')
    counts = data['user_id'].map(data.groupby('user_id')['ts'].apply(len))
    thrs_train = (ranks/counts) <= train_size
    thres_train = pd.DataFrame(thrs_train, columns=["thrs_train"])
    data = data.join(thres_train)
    X_train = data[data['thrs_train'] == True]
    #print("X_train: ", X_train)
    X_train = X_train.drop(columns=['thrs_train','ts'], axis=1)
    #print("X_train: ", X_train)
    if train_size + test_size == 1.0:
        X_test = data[data['thrs_train'] == False]
        #print("X_test: ", X_test)
        X_test = X_test.drop(columns=['thrs_train','ts'], axis=1)
        #print("X_test: ", X_test)
    else:
        thrs_test = (ranks/counts) <= (train_size + test_size)
        thres_test = pd.DataFrame(thrs_test, columns=["thrs_test"])
        data = data.join(thres_test)
        data["thrs"] = data["thrs_test"] > data["thrs_train"]
        X_test = data[data['thrs'] == True]
        #print("X_test: ", X_test)
        X_test = X_test.drop(columns=['thrs', 'thrs_train', 'thrs_test', 'ts'], axis=1)
        #print("X_test: ", X_test)
  
    X_train = X_train.applymap(str)
    X_test = X_test.applymap(str)
    X_train = X_train.sort_values(by=['user_id'])
    X_test = X_test.sort_values(by=['user_id'])
    return (X_train, X_test)


def prepare_attributes_for_classifier(user_info, users, attr_type):
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


### Recommender system functions ###

def generate_recommendations(X_train, X_test, user_features, users, use_features=True):
    # Build and train FM model
    rankfm = RankFM(factors=10, loss='bpr', max_samples=10, alpha=0.01, sigma=0.1, learning_rate=0.1, learning_schedule='invscaling')
    #rankfm = RankFM(factors=20, loss='warp', max_samples=20, alpha=0.01, sigma=0.1, learning_rate=0.1, learning_schedule='invscaling')
    if use_features == True:
        rankfm.fit(X_train, user_features=user_features, epochs=20, verbose=True)
    else:
        rankfm.fit(X_train, epochs=20, verbose=True)
    # Generate TopN Recommendations
    recommendations = rankfm.recommend(users, n_items=N, filter_previous=True, cold_start="nan")
    print("recommendations_train shape: ", recommendations.shape)

    # Generate Model Scores for Validation Interactions
    scores = rankfm.predict(X_test, cold_start="nan")
    print("Scores shape: ", scores.shape)
    print(pd.Series(scores).describe())
    return rankfm, recommendations, scores



### Classification functions ###

def classify(classifier, X_train, X_test, y_train, y_test):
    # X is recommendaitons for each user (943, 10)
    # z is true genders shape(943,1)

    pipe = make_pipeline(StandardScaler(), classifier)
    pipe.fit(X_train, y_train)
    results = {}
    results["y_true"] = y_test
    
    results["y_pred"] = pipe.predict(X_test)
    results["y_prob"] = pipe.predict_proba(X_test)
    results["score"] = pipe.score(X_test, y_test)
    print("Score: ", results["score"], "\n")

    #print("Y_test: ", y_test)
    #print("proba: ", results["y_prob"])
    return results


### Evaluation functions ###

def evaluate_recommender(model, X_test):
    # Evaluate model
    scores = {}
    scores["p"] = precision(model, X_test, k=N)
    scores["r"] = recall(model, X_test, k=N)
    scores["hr"] = hit_rate(model, X_test, k=N)

    print("precision: {:.3f}".format(scores["p"]))
    print("recall: {:.3f}".format(scores["r"]))
    print("hit rate: {:.3f}".format(scores["hr"]))

    return scores

def get_roc_auc_score(y_test, y_score):
    #Create one-hot encoding
    print("y_score shape: ", y_score.shape)
    print("y_score: ", y_score)
    y_test = pd.DataFrame(data=y_test)
    n_classes = y_score.shape[1]

    enc = OneHotEncoder()
    y_test = enc.fit_transform(y_test).toarray()
    
    print("y_test shape: ", y_test.shape)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    print("roc_auc: ",roc_auc["micro"])
    return roc_auc["micro"]



### Utility functions ###

# Get cold-start users and items
def get_coldstart_units(train_units, test_units, unit_name="units"):
    cold_start_units = set(test_units) - set(train_units)
    return cold_start_units

# Generates an output file name
def get_output_filename(base, result):
    filename = base
    if result["attr"] == "gender":
        filename += "_g"
    elif result["attr"] == "age":
        filename += "_a"
    elif result["attr"] == "occupation":
        filename += "_o"
    elif result["attr"] == "location":
        if LOC_TYPE == STATE:
            filename += "_s"
        elif LOC_TYPE == COUNTY:
            filename += "_t"
        elif LOC_TYPE == CITY:
            filename += "_c"
    filename += ".csv"
    return filename

# Turns recommendations into a matrix
def recs_to_matrix(recs):
    rec_matrix = [] 
    rec_matrix = np.zeros((NUM_USERS, NUM_ITEMS), dtype=np.double)
    row_index = []
    column_index = []
    for i in range(NUM_USERS):
        row_index.append(str(i+1))
    for j in range(NUM_ITEMS):
        column_index.append(str(j+1))
    
    for usr in range(recs.shape[0]):
        for rnk in range(recs.shape[1]):
            item = recs[rnk][usr]
            rec_matrix[int(usr)][int(item)] = 1.0
    rec_matrix = pd.DataFrame(data=rec_matrix, index=row_index, columns=column_index)
    return rec_matrix

# Returns classifier object
def get_classifier(clf_str, attr):
    classifier = None
    if clf_str == "dummy":
        classifier = DummyClassifier(strategy="most_frequent")
    elif clf_str == "log_reg":
        classifier = LogisticRegression(max_iter=500)
    elif clf_str == "svc":
        classifier = svm.SVC(probability=True)
    elif clf_str == "ran_for":
        classifier = RandomForestClassifier()
    return classifier

# Adds user attributes to recommendations
def add_attr_to_recs(recs, attr, attr_values):
    recs[attr] = attr_values
    return recs

# Returns users and items for given set
def get_users_items(x_train, x_test):
    train_users = np.sort(x_train.user_id.unique())
    test_users = np.sort(x_test.user_id.unique())
    train_items = np.sort(x_train.item_id.unique())
    test_items = np.sort(x_test.item_id.unique())
    return(train_users, train_items, test_users, test_items)



### Print functions ###

# Print Matrix Dimensions for training/test sets
def print_matrix_dim(x_set, set_name=""):
    print("Matrix Dimensions for ", set_name)
    print(set_name, " shape: {}".format(x_set.shape))
    print(set_name, " unique users: {}".format(x_set.user_id.nunique()))
    print(set_name, " unique items: {}".format(x_set.item_id.nunique()))
    print("\n")


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




### Write to csv functions ###

def write_double_rec_to_csv(rec_train, scores_train, rec_test,  scores_test):
    conf_scores_train = rec_train.copy()
    conf_scores_test = rec_test.copy()

    if N > 20:
        print("Recommendations are not written to file for N higher than 20.")
    else:

        with open('recomended_items.csv','w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(["User", "Item_tr",  "Item_te", "Rank", "Conf_score_tr",   "Conf_score_te"] )

        ind = 0

        for usr in range(rec_train.shape[0]):
            for rnk in range(rec_train.shape[1]):
                conf_scores_train[rnk][usr] = scores_train[ind]
                conf_scores_test[rnk][usr] = scores_test[ind]
                with open('recomended_items.csv','a', newline='') as csvfile:
                    csv_writer = csv.writer(csvfile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    csv_writer.writerow([ usr, rec_train[rnk][usr],  rec_test[rnk][usr],  rnk+1, conf_scores_train[rnk][usr],  conf_scores_test[rnk][usr]] )
                ind += 1

def write_rec_scores_to_csv(all_results):

    with open('rec_scores.csv','w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter='|', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(["Nr\t",  "P@K\t\t\t",  "R@K\t\t\t",  "HR\t\t\t"] )


    for round in all_results.keys():
        result = all_results[round]
        with open('rec_scores.csv','a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow([ round, result["p"],  result["r"],  result["hr"] ] )

def write_clf_scores_to_csv(all_results):

    with open('clf_scores.csv','w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(["Clf\t",  "Gender\t\t\t",  "Age\t\t\t",  "Job\t\t\t",  LOC_TYPE] )

    output = {}
    for result in all_results:
        clf = result["clf"]
        if clf not in output.keys():
            output[clf] = {}
            for attr in INFER_ATTR.keys():
                output[clf][attr] = "------------------"
        attr = result["attr"]
        output[clf][attr] = result["roc_auc"]

    for clf in output.keys():
        with open('clf_scores.csv','a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow([ clf, output[clf]["gender"],  output[clf]["age"],  output[clf]["occupation"],  output[clf]["location"] ] )

def write_clf_preds_to_csv(result):
    # Make a separate file for each clf - attr pair
    base = "output/prediction_values/clf_pred"
    filename = get_output_filename(base, result)
    #class_labels = ["c_" + x for x in range(len(result["y_prob"]))]

    with open(filename,'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(["User", "y_true", "y_pred"])
    for i in range(len(result["users"])):
        
        with open(filename,'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow([ result["users"][i], result["y_true"][i],  result["y_pred"][i] ] )








def main():
    print("Program starting... \n")

    # Legg til attributtene til ratings bortsett fra det som skal klassifiseres

    # Load user info
    user_info = load_user_data()
    #(user_info, user_features) = load_user_data()

    # Load user features
    user_features = load_user_features()
    

    # Load interaction data and create training and test sets
    interaction_data = load_interaction_data() 

    # Create train and test sets
    (X_train1, X_test1) = prepare_splits(interaction_data, train_size=0.4, test_size=0.3)
    (X_train2, X_test2) = prepare_splits(interaction_data, train_size=0.7, test_size=0.3)

    # Get train and test users
    (train_users1, train_items1, test_users1, test_items1) = get_users_items(X_train1, X_test1)
    (train_users2, train_items2, test_users2, test_items2) = get_users_items(X_train2, X_test2)

    # Training and test set dimensions
    print_matrix_dim(X_train1, "X_train1")
    evaluate_matrix_sparsity(X_train1, "X_train1")

    print_matrix_dim(X_test1, "X_test1")
    evaluate_matrix_sparsity(X_test1, "X_test1")

    print_matrix_dim(X_train2, "X_train2")
    evaluate_matrix_sparsity(X_train2, "X_train2")

    print_matrix_dim(X_test2, "X_test2")
    evaluate_matrix_sparsity(X_test2, "X_test2")

    
    # User and Item stats
    print("X train and test 1")
    print_user_item_stats(train_users1, test_users1, "users")
    print_user_item_stats(train_items1, test_items1, "items")
    print("X train and test 2")
    print_user_item_stats(train_users2, test_users2, "users")
    print_user_item_stats(train_items2, test_items2, "items")
    
    #print(user_features)

    # Generate recommendations_train
    rec_scores = {}
    print("Recommender Round 1: ")
    rankfm1, recommendations_train, scores_train = generate_recommendations(X_train1, X_test1, user_features, train_users1, use_features=True)
    rec_scores["round1"] = evaluate_recommender(rankfm1, X_test1)

    # Generate recommendations_test
    print("Recommender Round 2: ")
    rankfm2, recommendations_test, scores_test = generate_recommendations(X_train2, X_test2, user_features, train_users2, use_features=True)
    rec_scores["round2"] = evaluate_recommender(rankfm2, X_test2)

    #Write recommendation results to file
    write_double_rec_to_csv(recommendations_train, scores_train, recommendations_test, scores_test)
    write_rec_scores_to_csv(rec_scores)


    # Classification
    attributes_train = {}
    attributes_test = {}
    classifier = None
    all_score_results = []
    
    rec_train = recs_to_matrix(recommendations_train)
    rec_test = recs_to_matrix(recommendations_test)
    print("Rec train shape: ", rec_train.shape)
    print("Rec test shape: ", rec_test.shape)

    for attr in INFER_ATTR.keys():
        if INFER_ATTR[attr] == True:
            # Prepare gender attributes for classification
            attributes_train[attr] = prepare_attributes_for_classifier(user_info, test_users1, attr_type=attr)
            attributes_test[attr] = prepare_attributes_for_classifier(user_info, test_users2, attr_type=attr)
            #print(attributes_test[attr])
    
    for attr in INFER_ATTR.keys():
        if INFER_ATTR[attr] == True:
            recs_train_ctx = copy.deepcopy(rec_train)
            recs_test_ctx  = copy.deepcopy(rec_test)
            
            #print("recs before: ",  recs_train_ctx )
            #print(attr, " attributes train len: ", len(attributes_train[attr]))
            #print(attr, " attributes test len: ", len(attributes_test[attr]))

            for attr2 in INFER_ATTR.keys():
                if INFER_ATTR[attr2] == True and attr != attr2:
                    recs_train_ctx = add_attr_to_recs(recs_train_ctx, attr2, attributes_train[attr2])
                    recs_test_ctx = add_attr_to_recs(recs_test_ctx, attr2, attributes_test[attr2])
            #print("recs: ", recs_train_ctx)

            # Classify attribute
            for clf in CLASSIFIERS.keys():
                if CLASSIFIERS[clf] == True:
                    
                    print("## ", clf, " for ", attr, " ##")
                    classifier = get_classifier(clf, attr)
                    score_results = classify(classifier, recs_train_ctx, recs_test_ctx, attributes_train[attr], attributes_test[attr])
                    score_results ["attr"] = attr
                    score_results ["clf"] = clf
                    score_results["users"] = test_users2
                    all_score_results.append(score_results)
                    score_results["roc_auc"] = get_roc_auc_score(attributes_test[attr], score_results["y_prob"])
                    write_clf_preds_to_csv(score_results)
                    #write_clf_probs_to_csv(score_results)
            
    write_clf_scores_to_csv(all_score_results)

    print("Males: ", str(statistics["males"] /943))
    print("Age under 35: ", str(statistics["age0"] /943))
    print("Age under 45: ", str(statistics["age1"] /943))
    print("Age over 45: ", str(statistics["age2"] /943))



if __name__ == "__main__":
    main()








"""
def prepare_attributes_for_recommender(user_info,  user_features):
    attr_classes = {}
    for attr_type in INFER_ATTR.keys():
        if INFER_ATTR[attr_type] == True:
            attr_classes[attr_type] = {}

    def is_in_age_group(age, age_cat):
        return True if age >= AGE_GROUPS[age_cat][0] and age <= AGE_GROUPS[age_cat][1] else False
    
    # Map zip code to state/city/county
    def map_location(zipcode, loc_type):
        search = SearchEngine(simple_zipcode=True)
        zip_code = search.by_zipcode(zipcode)
        zip_code = zip_code.to_dict()
        return zip_code[loc_type]

    
    for usr_id in range(len(user_info)):
        for attr_type in INFER_ATTR.keys():
            if INFER_ATTR[attr_type] == True:
                attr_value = user_info[usr_id+1][attr_type]

                if attr_type == "age":
                    for age_cat in AGE_GROUPS.keys():
                        if is_in_age_group(attr_value, age_cat):
                            new_attr_value = age_cat
                else:
                    if attr_type == "location":
                        attr_value = map_location(attr_value, LOC_TYPE)
                    # Create dict of attribute labels
                    if attr_value not in attr_classes[attr_type].keys():
                        attr_classes[attr_type][attr_value] = len(attr_classes[attr_type])
                    new_attr_value = attr_classes[attr_type][attr_value]

                # Create array of attribute representations
                user_features[usr_id][attr_type] = new_attr_value
    for cls in attr_classes.keys():
        print(attr_classes[cls])
    user_features = pd.DataFrame(data=user_features)
    return user_features

def write_clf_probs_to_csv(result):
    # Make a separate file for each clf - attr pair
    
    base = "output/prediction_probabilities/clf_prob"
    filename = get_output_filename(base, result)
    class_labels = []
    attr = result["attr"]
    for i in range(len(result["y_prob"][0])):
        class_prob = attr + str(i)
        class_labels.append("c_" + str(i))

    with open(filename,'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(class_labels)

    for i in range(len(result["users"])):
        with open(filename,'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(result["y_prob"] )

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

"""

