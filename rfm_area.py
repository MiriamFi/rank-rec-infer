# Imports
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from uszipcode import SearchEngine
import collections

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
from sklearn import svm, preprocessing
from sklearn.dummy import DummyClassifier

# Sklearn metric imports
from sklearn.metrics import roc_curve, auc, accuracy_score, roc_auc_score, f1_score, log_loss

# Sklearn pipeline imports
from sklearn.pipeline import  Pipeline
from sklearn.preprocessing import StandardScaler

# Sklearn CV and hyperparameter tuning imports
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold

from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix
from sklearn.metrics import plot_roc_curve
from sklearn.preprocessing import label_binarize

from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import label_binarize
# Constants
N = 50

K_OUTER = 5
K_INNER = 3

INFER_ATTR = {
    "gender": True,
    "age": True,
    "occupation": True,
    "state": False,
    "county": False,
    "city": False,
    'area_5' : False,
    'area_2' : True
}

INCLUDE_FEATURES = {
    "gender": True,
    "age": True,
    "occupation": True,
    "state": False,
    "county": False,
    "city": False,
    'area_5' : False,
    'area_2' : True
}



USZ_NAMES = {
    "state": "state",
    "county": "county",
    "city": "major_city"
}

STATE_AREA_5 = {
    'west' : ['WA', 'OR', 'ID', 'MT', 'WY', 'CO', 'UT', 'NV', 'CA', 'AK', 'HI'],
    'midwest' : ['ND', 'SD', 'NE', 'KS', 'MN', 'IA', 'MO', 'WI', 'IL', 'IN', 'MI', 'OH'],
    'southwest' : ['AZ', 'NM', 'OK', 'TX'],
    'northeast' : ['NY', 'PA', 'NJ', 'CT', 'RI', 'MA', 'NH', 'ME', 'VT'],
    'southeast' : ['AR', 'LA', 'MS', 'AL', 'GA', 'FL', 'SC', 'NC', 'VA', 'DC', 'DE','MD', 'WV', 'KY','TN'],
    'none' : None
}

STATE_AREA_2 = {
    'west' : ['ND', 'SD', 'NE', 'KS', 'WA', 'OR', 'ID', 'MT', 'WY', 'CO', 'UT', 'NV', 'CA', 'AK', 'HI', 'AZ', 'NM', 'OK', 'TX'],
    'east' : ['MN', 'IA', 'MO', 'WI', 'IL', 'IN', 'MI', 'OH', 'NY', 'PA', 'NJ', 'CT', 'RI', 'MA', 'NH', 'ME', 'VT', 'AR', 'LA', 'MS', 'AL', 'GA', 'FL', 'SC', 'NC', 'VA', 'DC', 'DE','MD', 'WV', 'KY','TN'],
    'none' : None
}

AREA_CAT_5 = {
    'west' : 0,
    'midwest' : 1,
    'southwest' : 2,
    'northeast' : 3,
    'southeast' : 4,
    'none' : 5
}

AREA_CAT_2 = {
    'west' : 0,
    'east': 1,
    'none' : 2
}

OCCUPATIONS = {
    'technician': 'technician/engineer',
    'lawyer': 'lawyer',
    'executive': 'executive',
    'student': 'student',
    'programmer': 'programmer',
    'engineer': 'technician/engineer',
    'retired': 'homemaker/retired',
    'scientist': 'scientist',
    'educator': 'educator',
    'other': 'other/none',
    'salesman': 'salesman/marketing',
    'healthcare': 'healthcare/doctor',
    'administrator': 'administrator',
    'librarian': 'librarian',
    'writer': 'writer',
    'artist': 'artist',
    'none': 'other/none',
    'marketing': 'salesman/marketing',
    'doctor': 'healthcare/doctor',
    'entertainment': 'entertainment',
    'homemaker': 'homemaker/retired'
    }


AGE_GROUPS = {
    0: [0, 34],
    1: [35, 45],
    2: [46, 99]
}

CLASSIFIERS = {
    "dummy": True,
    "log_reg": True,
    "svc": True,
    "ran_for": True
}

RAN_FOR_HPARAMS = {
    "ran_for__n_estimators": [100],
    "ran_for__max_features": [2],
    # 'ran_for__bootstrap': [True, False],
    # 'ran_for__min_samples_leaf': [1, 2, 4],
    # 'ran_for__min_samples_split': [2, 5, 10],
    # 'ran_for__max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
}

LOG_REG_HPARAMS = {
    'log_reg__solver': ['lbfgs'],
    'log_reg__C': [10],
    'log_reg__penalty': ['l2'],
    'log_reg__max_iter' : [1000]
}

SVC_HPARAMS = {
    'svc__kernel': ['linear'],
    'svc__C': [10],
    'svc__probability' : [True]
}

DUMMY_HPARAMS ={
    'dummy__strategy' : ["most_frequent"]
}

CLF_HPARAMS = {
    "dummy": DUMMY_HPARAMS,
    "log_reg": LOG_REG_HPARAMS,
    "svc": SVC_HPARAMS,
    "ran_for": RAN_FOR_HPARAMS}

CLF_PLOT_NAME = {
    "dummy": 'MFC',
    "log_reg": 'LGC',
    "svc": 'SVC',
    "ran_for": 'RFC'}

NUM_USERS = 943
NUM_ITEMS = 1682


# Variales


### Data Loader functions ###

# Load interaction data
def load_interaction_data(filename="u.data", path="data/ml-100k/"):
    data = []  # user id + movie id
    with open(path + filename) as f:
        for line in f:
            (user, movieid, rating, ts) = line.split('\t')
            data.append({"user_id": int(user), "item_id": int(movieid), "ts": int(ts.strip())})

    # Prepare data
    data = pd.DataFrame(data=data)
    # print(data.iloc[0])
    return data


# load user data
def load_user_data(filename="u.user", path="data/ml-100k/"):
    user_info = {}
    # user_features = []
    with open(path + filename, 'r') as fin:
        for line in fin.readlines():
            user_id, age, gender, occu, zipcode = line.split('|')
            user_info[int(user_id)] = {
                'age': int(age),
                'gender': gender,
                'occupation': occu,
                'state': str(zipcode).strip(),
                'county': str(zipcode).strip(),
                'city': str(zipcode).strip(),
                'area_5' : str(zipcode).strip(),
                'area_2' : str(zipcode).strip()
            }
            # user_features.append({"user_id": str(user_id)})
        print('User Info Loaded!')
    return user_info
    # return (user_info, user_features)


# Load user features
def load_user_features():
    # Gets filename for user features
    filename = "user_features/ml_new/feat"
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
    elif INCLUDE_FEATURES["area_5"] == True:
        filename += "_r5"
    elif INCLUDE_FEATURES["area_2"] == True:
        filename += "_r2"
    filename += ".csv"

    usr_feat = pd.read_csv(filename)
    print("Loaded user features from", filename)
    usr_feat = usr_feat.astype(str)
    print("User features shape: ", usr_feat.shape, "\n")
    return usr_feat


### Data preparation functions ###

def prepare_rec_splits(data, train_size=0.9, test_size=0.1):
    ranks = data.groupby('user_id')['ts'].rank(method='first')
    counts = data['user_id'].map(data.groupby('user_id')['ts'].apply(len))
    thrs_train = (ranks / counts) <= train_size
    thres_train = pd.DataFrame(thrs_train, columns=["thrs_train"])
    data = data.join(thres_train)
    X_train = data[data['thrs_train'] == True]
    # print("X_train: ", X_train)
    X_train = X_train.drop(columns=['thrs_train', 'ts'], axis=1)
    # print("X_train: ", X_train)
    if train_size + test_size == 1.0:
        X_test = data[data['thrs_train'] == False]
        # print("X_test: ", X_test)
        X_test = X_test.drop(columns=['thrs_train', 'ts'], axis=1)
        # print("X_test: ", X_test)
    else:
        thrs_test = (ranks / counts) <= (train_size + test_size)
        thres_test = pd.DataFrame(thrs_test, columns=["thrs_test"])
        data = data.join(thres_test)
        data["thrs"] = data["thrs_test"] > data["thrs_train"]
        X_test = data[data['thrs'] == True]
        # print("X_test: ", X_test)
        X_test = X_test.drop(columns=['thrs', 'thrs_train', 'thrs_test', 'ts'], axis=1)
        # print("X_test: ", X_test)

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

    
    # Map zip code to state
    def map_location(zipcode, attr_type):
        search = SearchEngine(simple_zipcode=True)
        zip_code = search.by_zipcode(zipcode)
        zip_code = zip_code.to_dict()
        area = map_state_to_area(zip_code['state'], attr_type)
        return area
    
    def map_state_to_area(state, attr_type):
        if attr_type == 'area_5':
            for key in STATE_AREA_5.keys():
                if state == None:
                    return 'none'
                elif state in STATE_AREA_5[key]:
                    return key
        elif attr_type == 'area_2':
            for key in STATE_AREA_2.keys():
                if state == None:
                    return 'none'
                elif state in STATE_AREA_2[key]:
                    return key
            

    for usr_id in new_user_info.keys():
        attr_value = new_user_info[usr_id][attr_type]

        if attr_type == "age":
            for age_cat in AGE_GROUPS.keys():
                if is_in_age_group(attr_value, age_cat):
                    new_attr_value = age_cat
        elif attr_type == "area_5":
                attr_value = map_location(attr_value, attr_type)
                new_attr_value = AREA_CAT_5[attr_value]
        elif attr_type == "area_2":
                attr_value = map_location(attr_value, attr_type)
                new_attr_value = AREA_CAT_2[attr_value]
        else:
            if attr_type == 'occupation':
                attr_value = OCCUPATIONS[attr_value]
            # Create dict of attribute labels
            if attr_value not in attr_classes.keys():
                attr_classes[attr_value] = len(attr_classes)
            new_attr_value = attr_classes[attr_value]

        # Create array of attribute representations
        attributes.append(new_attr_value)
    print("atr_classes: ", attr_classes)
    return attributes


### Recommender system functions ###

def generate_recommendations(X_train, X_test, user_features, users, use_features=True):
    # Build and train FM model
    rankfm = RankFM(factors=15, loss='bpr', learning_schedule='constant')
    #rankfm = RankFM(factors=10, loss='bpr', max_samples=10, alpha=0.01, sigma=0.1, learning_rate=0.1,learning_schedule='invscaling')

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


def cross_validate(clf, X_r1, X_r2, y_r1, y_r2, attr):
    # configure the cross-validation procedure
    classes = y_r1 + y_r2
    #print("len classes: ", len(classes))
    classes = np.unique(classes)
    #print("y_r1: ", y_r1)
    #print("len classes: ", len(classes))
    #print("classes: ", classes)
    #k_outer = len(y_r1) - 10
    cv_outer = KFold(n_splits=K_OUTER, shuffle=True, random_state=1)


    # enumerate splits
    outer_results_acc = list()
    outer_results_f1 = list()
    outer_results_auc = list()

    #loo = LeaveOneOut()
    #loo.get_n_splits(X_r1)
    for train_ix, test_ix in cv_outer.split(X_r1):

    #for train_ix, test_ix in cv_outer.split(X_r1):
        #print("train_ix: ",train_ix)
        #print("test_ix: ",test_ix)

        # split data
        train_ix_str = [str(x + 1) for x in train_ix]
        test_ix_str = [str(x + 1) for x in test_ix]
        X_train = X_r1[X_r1.index.isin(train_ix_str)]
        X_test = X_r2[X_r2.index.isin(test_ix_str)]

        y_r1 = np.array(y_r1)
        y_r2 = np.array(y_r2)
        y_train, y_test = y_r1[train_ix], y_r2[test_ix]

        #print("y_train len: ", len(y_train))
        #print("y_test len: ", len(y_test))
        n_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))
        #print("y_train unique: ", np.unique(y_train))
        print("y_train unique len: ", len(np.unique(y_train)))
        #print("y_test unique: ", np.unique(y_test))
        print("y_test unique len: ", len(np.unique(y_test)))
        #print("y all unique: ", np.unique(np.concatenate((y_train, y_test), axis=0)))
        #print("y all unique len: ", len(np.unique(np.concatenate((y_train, y_test), axis=0))))
        print(set(y_train) - set(y_test))

        # Normalize the data
        X_train = preprocessing.normalize(X_train, norm='l2')
        X_test = preprocessing.normalize(X_test, norm='l2')

        # configure the cross-validation procedure
        #cv_inner = KFold(n_splits=K_INNER, shuffle=True, random_state=1)
        cv_inner = KFold(n_splits=K_INNER, shuffle=True, random_state=1)
        classifier = get_classifier(clf)
        pipe = Pipeline(steps=[('scaler', StandardScaler()), (clf, classifier)]) # ('scaler', StandardScaler()) ('t', trans)

        # define search space and search
        space = CLF_HPARAMS[clf]

        search = GridSearchCV(pipe, space, scoring='accuracy', cv=cv_inner, refit=True)

        # execute search
        result = search.fit(X_train, y_train)

        # get the best performing model fit on the whole training set
        best_model = result.best_estimator_

        # evaluate model on the hold out dataset
        y_pred = best_model.predict(X_test)

        # evaluate the model
        acc = accuracy_score(y_test, y_pred)

        clf_classes = best_model[clf].classes_
        f1 = f1_score(y_test, y_pred, average='micro', labels=clf_classes)


        #y_test = label_binarize(y_test, classes=classes)
        #viz = plot_roc_curve(best_model, X_test, y_test, pos_label=1)
        #auc_score = viz.roc_auc
        #print(auc_score)

        


        clf_classes = best_model[clf].classes_
        #print("classes: ", clf_classes)
        y_prob = search.predict_proba(X_test)
        auc_score = plot_roc_auc(y_test, y_prob, clf, attr, classes)
        #auc_score = get_roc_auc( y_test, y_prob)
        #auc_score = plot_roc_auc(y_test, y_prob, clf, attr)
        #print("auc: ", auc_score)
        
        # store the result
        outer_results_acc.append(acc)
        outer_results_f1.append(f1)
        outer_results_auc.append(auc_score)
        #outer_results_auc.append(auc_score)

        # report progress
        print('>acc=%.3f, f1=%.3f, auc=%.3f, est=%.3f, cfg=%s' % (acc, f1, auc_score, result.best_score_, result.best_params_), "\n")

    # summarize the estimated performance of the model
    mean_acc = np.mean(outer_results_acc)
    std_acc = np.std(outer_results_acc)
    print('Accuracy: %.3f (%.3f)' % (mean_acc, std_acc), "\n")
    mean_f1 = np.mean(outer_results_f1)
    std_f1 = np.std(outer_results_f1)
    print('F1: %.3f (%.3f)' % (mean_f1, std_f1), "\n")
    mean_auc = np.mean(outer_results_auc)
    std_auc = np.std(outer_results_auc)
    print('AUC: %.3f (%.3f)' % (mean_auc, std_auc), "\n")


    return (mean_acc, mean_f1, mean_auc)


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

def plot_roc_auc(y_test, y_prob, clf, attr, classes):
    

    y_test = np.array(y_test)
    y_test = label_binarize(y_test, classes=classes)

    if len(classes) == 2:
        fpr, tpr, thresholds = roc_curve(y_test, y_prob[:, 1], pos_label=1)
    else: 
        fpr, tpr, thresholds = roc_curve(y_test[:, 1], y_prob[:, 1], pos_label=1)

    clf_name = CLF_PLOT_NAME[clf]
    fig = plt.figure(figsize=(8, 4))
    # plt.plot(fpr_LR, tpr_LR, linestyle='-', label='Log. Reg.')
    plt.plot(fpr, tpr, linestyle='-.', lw=2, label=clf_name)
    plt.legend()
    auc_score = auc(x=fpr, y=tpr)
    plt.title(clf_name + ' AUC: {:.3f}'.format( auc(x=fpr, y=tpr)), fontsize=14)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    #plt.show()
    filename = "output/plots/auc_"  + attr + "_" + clf
    plt.savefig(filename)
    return auc_score
 

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
    elif result["attr"] == "state":
        filename += "_s"
    elif result["attr"] == "county":
        filename += "_t"
    elif result["attr"] == "city":
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
        row_index.append(str(i + 1))
    for j in range(NUM_ITEMS):
        column_index.append(str(j + 1))

    for usr in range(recs.shape[0]):
        for rnk in range(recs.shape[1]):
            item = recs[rnk][usr]
            rec_matrix[int(usr)][int(item)] = 1.0
    rec_matrix = pd.DataFrame(data=rec_matrix, index=row_index, columns=column_index)
    return rec_matrix


# Returns classifier object
def get_classifier(clf_str):
    classifier = None
    if clf_str == "dummy":
        classifier = DummyClassifier()
    elif clf_str == "log_reg":
        classifier = LogisticRegression()
    elif clf_str == "svc":
        classifier = svm.SVC()
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
    return (train_users, train_items, test_users, test_items)


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


# Evaluate X_train Matrix Sparsity
def evaluate_matrix_sparsity(x_set, set_name=""):
    unique_users = x_set.user_id.nunique()
    unique_items = x_set.item_id.nunique()
    sparsity = 1 - (len(x_set) / (unique_users * unique_items))
    print(set_name, " matrix sparsity: {}%".format(round(100 * sparsity, 1)))
    print("\n")


### Write to csv functions ###

def write_double_rec_to_csv(rec_train, scores_train, rec_test, scores_test):
    conf_scores_train = rec_train.copy()
    conf_scores_test = rec_test.copy()

    if N > 20:
        print("Recommendations are not written to file for N higher than 20.")
    else:

        with open('recomended_items.csv', 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(["User", "Item_tr", "Item_te", "Rank", "Conf_score_tr", "Conf_score_te"])

        ind = 0

        for usr in range(rec_train.shape[0]):
            for rnk in range(rec_train.shape[1]):
                conf_scores_train[rnk][usr] = scores_train[ind]
                conf_scores_test[rnk][usr] = scores_test[ind]
                with open('recomended_items.csv', 'a', newline='') as csvfile:
                    csv_writer = csv.writer(csvfile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    csv_writer.writerow(
                        [usr, rec_train[rnk][usr], rec_test[rnk][usr], rnk + 1, conf_scores_train[rnk][usr],
                         conf_scores_test[rnk][usr]])
                ind += 1


def write_rec_scores_to_csv(all_results):
    with open('rec_scores.csv', 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter='|', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(["Nr\t", "P@K\t\t\t", "R@K\t\t\t", "HR\t\t\t"])

    for round in all_results.keys():
        result = all_results[round]
        with open('rec_scores.csv', 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow([round, result["p"], result["r"], result["hr"]])


def write_clf_scores_to_csv(all_results, metric):
    filename = 'clf_' + metric + '_score.csv'
    with open(filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(
            ["Clf\t", "Gender\t\t\t", "Age\t\t\t", "Job\t\t\t", "Area_5\t\t\t", "Area_2\t\t\t", ])

    output = {}
    for result in all_results:
        clf = result["clf"]
        if clf not in output.keys():
            output[clf] = {}
            for attr in INFER_ATTR.keys():
                output[clf][attr] = "------------------"
        attr = result["attr"]
        output[clf][attr] = result[metric]

    for clf in output.keys():
        with open(filename, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(
                [clf, output[clf]["gender"], output[clf]["age"], output[clf]["occupation"], output[clf]["area_5"],
                 output[clf]["area_2"]])


def write_clf_preds_to_csv(result):
    # Make a separate file for each clf - attr pair
    base = "output/prediction_values/clf_pred"
    filename = get_output_filename(base, result)
    # class_labels = ["c_" + x for x in range(len(result["y_prob"]))]

    with open(filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(["User", "y_true", "y_pred"])
    for i in range(len(result["users"])):
        with open(filename, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow([result["users"][i], result["y_true"][i], result["y_pred"][i]])

def count_attr_samples(count_attr, limit, less_than):
    num_of_target_samples = 0
    for key in count_attr.keys():
        if less_than==True:
            if count_attr[key] <= limit:
                num_of_target_samples += 1
        else:
            if count_attr[key] == limit:
                num_of_target_samples += 1
    return num_of_target_samples

def main():
    print("Program starting... \n")

    # Legg til attributtene til ratings bortsett fra det som skal klassifiseres

    # Load user info
    user_info = load_user_data()

    # Load user features
    user_features = load_user_features()

    # Load interaction data and create training and test sets
    interaction_data = load_interaction_data()
    users = np.sort(interaction_data.user_id.unique())

    # Create train and test sets
    (X_train1, X_test1) = prepare_rec_splits(interaction_data, train_size=0.4, test_size=0.3)
    (X_train2, X_test2) = prepare_rec_splits(interaction_data, train_size=0.7, test_size=0.3)

    # Get train and test users
    (train_users1, train_items1, test_users1, test_items1) = get_users_items(X_train1, X_test1)
    (train_users2, train_items2, test_users2, test_items2) = get_users_items(X_train2, X_test2)

    """
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
    print_user_item_stats(train_items2, test_items2, "items")"""

    # print(user_features)
    
    # Generate recommendations_train
    rec_scores = {}
    print("Recommender Round 1: ")
    rankfm1, recommendations_train, scores_train = generate_recommendations(X_train1, X_test1, user_features,
                                                                            train_users1, use_features=True)
    rec_scores["round1"] = evaluate_recommender(rankfm1, X_test1)

    # Generate recommendations_test
    print("Recommender Round 2: ")
    rankfm2, recommendations_test, scores_test = generate_recommendations(X_train2, X_test2, user_features,
                                                                          train_users2, use_features=True)
    rec_scores["round2"] = evaluate_recommender(rankfm2, X_test2)

    # Write recommendation results to file
    write_double_rec_to_csv(recommendations_train, scores_train, recommendations_test, scores_test)
    write_rec_scores_to_csv(rec_scores)
    
    # Classifier
    # X = interaction_data + extra attributes
    # y = true value for attribute

    # Classification
    attributes_train = {}
    attributes_test = {}

    rec_train = recs_to_matrix(recommendations_train)
    rec_test = recs_to_matrix(recommendations_test)
    print("Rec train shape: ", rec_train.shape)
    print("Rec test shape: ", rec_test.shape)

    results = []

    for attr in INFER_ATTR.keys():
        if INFER_ATTR[attr] == True:
            # Prepare gender attributes for classification
            attributes_train[attr] = prepare_attributes_for_classifier(user_info, test_users1, attr_type=attr)
            attributes_test[attr] = prepare_attributes_for_classifier(user_info, test_users2, attr_type=attr)
            #print("Diff in users: ", get_coldstart_units(test_users1, test_users1, unit_name="users"))
            # print(attributes_test[attr])


            print(attr)
            
            all_attr = np.concatenate((attributes_train[attr], attributes_test[attr]), axis=0)
            print(len(all_attr))
            counter_attr = collections.Counter(attributes_train[attr] )
            print("all_attr: ", counter_attr )
            
            print("all_attr unique: ", len(np.unique(all_attr)))
            print("less than 2 samples: ", count_attr_samples(counter_attr, 1, False) )
            print("2 samples: ", count_attr_samples(counter_attr, 2, False) )
            print("5 samples or less: ", count_attr_samples(counter_attr, 5, True) )
            #print("train_attr: ", collections.Counter(attributes_train[attr]))
            print("train_attr unique: ", len(np.unique(attributes_train[attr])))
            #print("test_attr: ", collections.Counter(attributes_test[attr]))
            print("test_attr unique: ", len(np.unique(attributes_test[attr])))
            print("\n\n")
            

    for attr in INFER_ATTR.keys():
        if INFER_ATTR[attr] == True:
            recs_train_ctx = copy.deepcopy(rec_train)
            recs_test_ctx = copy.deepcopy(rec_test)


            for attr2 in INFER_ATTR.keys():
                if INFER_ATTR[attr2] == True and attr != attr2:
                    recs_train_ctx = add_attr_to_recs(recs_train_ctx, attr2, attributes_train[attr2])
                    recs_test_ctx = add_attr_to_recs(recs_test_ctx, attr2, attributes_test[attr2])

            for clf in CLASSIFIERS.keys():
                if CLASSIFIERS[clf] == True:
                    print('\n')
                    print("CV with ", clf, " for ", attr)
                    result = {}
                    result['clf'] = clf
                    result['attr'] = attr
                    (result['acc'], result['f1'], result['auc']) = cross_validate(clf, recs_train_ctx, recs_test_ctx, attributes_train[attr], attributes_test[attr], attr)
                    results.append(result)
    write_clf_scores_to_csv(results, 'acc')
    write_clf_scores_to_csv(results, 'f1')
    write_clf_scores_to_csv(results, 'auc')



if __name__ == "__main__":
    main()
