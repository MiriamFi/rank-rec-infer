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

from sklearn.model_selection import GroupShuffleSplit
# Constants
N = 50

K_OUTER = 5
K_INNER = 3

INFER_ATTR = {
    "country": False,
    "usa_vs_rest" : True,
    "age": False
}

INCLUDE_FEATURES = {
    "country": True,
    "usa_vs_rest" : False,
    "age": False
}


OUTPUT_PATH = 'output_ex2_bx_03_c/'


STATE_AREA_5 = {
    'west' : ['WA', 'OR', 'ID', 'MT', 'WY', 'CO', 'UT', 'NV', 'CA', 'AK', 'HI'],
    'midwest' : ['ND', 'SD', 'NE', 'KS', 'MN', 'IA', 'MO', 'WI', 'IL', 'IN', 'MI', 'OH'],
    'southwest' : ['AZ', 'NM', 'OK', 'TX'],
    'northeast' : ['NY', 'PA', 'NJ', 'CT', 'RI', 'MA', 'NH', 'ME', 'VT'],
    'southeast' : ['AR', 'LA', 'MS', 'AL', 'GA', 'FL', 'SC', 'NC', 'VA', 'DC', 'DE','MD', 'WV', 'KY','TN'],
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

USA_VS_REST_CAT = {
    'usa' : 0,
    'rest': 1
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
    "ran_for__n_estimators": [100, 250],
    "ran_for__max_features": [10,20],
    'ran_for__min_samples_leaf': [ 2, 4],
    'ran_for__min_samples_split': [ 10,  20],
    'ran_for__random_state' : [16]
}

LOG_REG_HPARAMS = {
    'log_reg__solver' : [ 'newton-cg'],
    'log_reg__C' : [0.1, 0.01, 0.001],
    'log_reg__penalty' : ['l2'],
    'log_reg__max_iter' : [10000],
    'log_reg__random_state' : [16]
}

SVC_HPARAMS = {
    'svc__kernel': ['poly', 'rbf'],
    'svc__C': [ 1.0,  0.1, 0.01],
    'svc__probability' : [True],
    'svc__random_state' : [16]
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

NUM_USERS = 3839
NUM_ITEMS = 1682 #TODO

BX_SIZE = '0_3'

BX_SIZES = ['0_1', '0_3', '0_5', '0_7']
# Variales


### Data Loader functions ###


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
    elif INCLUDE_FEATURES["usa_vs_rest"] == True:
        filename += "_usa_" + BX_SIZE
    filename += ".csv"

    usr_feat = pd.read_csv(filename)
    print("Loaded user features from", filename)
    usr_feat = usr_feat.astype(str)
    print("User features shape: ", usr_feat.shape, "\n")
    return usr_feat


### Data preparation functions ###

def prepare_rec_splits(data, train_size=0.7, test_size=0.3):   
    #group_ids = [group_ids.get_group(x) for x in group_ids.groups]

    #TODO: shuffle the dataset
    data.sample(frac=1)

    data = data.sample(frac=1).reset_index(drop=True)
    #print(data)

    data['ranks'] = data.groupby('user_id').cumcount()
    data['ranks'] = data['ranks'].transform(lambda x: x+1)
    #print(data['ranks'])
    data['counts'] = data.groupby(["user_id"])["user_id"].transform("count")
    #print(data['counts'])
    data['thrs_train'] = (data['ranks'] / data['counts']) <= train_size
    #thres_train = pd.DataFrame(thrs_train, columns=["thrs_train"])
    #data = data.join(thres_train)
    #print("data: ", data)
    X_train = data[data['thrs_train'] == True]
    #print("X_train: ", X_train)
    X_train = X_train.drop(columns=['thrs_train', 'book_rating', 'ranks', 'counts'], axis=1)
    
    if train_size + test_size == 1.0:
        X_test = data[data['thrs_train'] == False]
        #print("X_test: ", X_test)
        X_test = X_test.drop(columns=['thrs_train', 'book_rating', 'ranks', 'counts'], axis=1)
        # print("X_test: ", X_test)
    else:
        data['thrs_test'] = (data['ranks'] / data['counts'] ) <= (train_size + test_size)
        data["thrs"] = data["thrs_test"] > data["thrs_train"]
        X_test = data[data['thrs'] == True]
        #print("X_test: ", X_test)
        X_test = X_test.drop(columns=['thrs', 'thrs_train', 'thrs_test', 'book_rating','ranks', 'counts'], axis=1)
        # print("X_test: ", X_test)

    #X_train = X_train.applymap(str)
    #X_test = X_test.applymap(str)
    X_train = X_train.sort_values(by=['user_id'])
    X_test = X_test.sort_values(by=['user_id'])
    return (X_train, X_test)


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


def cross_validate(clf, X_r1, X_r2, y_r1, y_r2, attr, df_users):
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

    df_users = df_users.sort_values(by=['user_id'])
    users = df_users['user_id'].to_list()


    for train_ix, test_ix in cv_outer.split(X_r1):

    #for train_ix, test_ix in cv_outer.split(X_r1):
        print("train_ix: ", train_ix)
        print("train_ix type: ", train_ix.dtype)
        print("test_ix: ", test_ix)
        print("X_r1.index: ", X_r1.index)
        print("X_r1.index types: ", X_r1.index.dtype) 


        

        # split data
        train_ix_str = [users[x] for x in train_ix]
        test_ix_str = [users[x] for x in test_ix]
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
    filename = OUTPUT_PATH + "plots/auc_"  + attr + "_" + clf
    plt.savefig(filename)
    plt.close()
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
    elif result["attr"] == "country":
        filename += "_country"
    elif result["attr"] == "usa_vs_rest":
        filename += "_usa_" + BX_SIZE

    filename += ".csv"
    return filename


# Turns recommendations into a matrix
def recs_to_matrix(recs,  isbn_map):
    print("recs: ", recs)
    


    matrix = np.zeros((recs.shape[0], len(isbn_map.keys())), dtype=np.double)
    cnt = 0
    for ind in recs.index:
        for i in range(N):
            matrix[cnt][isbn_map[recs[i][ind]]] = 1
        cnt += 1
    #for ind in range(recs.shape[0]):
        #for index in recs.index:
            #matrix[ind][isbn_map[recs[index][ind]]] =1
    for j in range(N):
        recs.drop([j], axis=1, inplace=True)
    for ind in range(len(isbn_map.keys())):
        recs[ind] = matrix[:,ind]
    

    print("recs: ", recs)
    print("recs shape: ", recs.shape)
    return recs
    """
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
    return rec_matrix"""


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
    train_items = np.sort(x_train.isbn.unique())
    test_items = np.sort(x_test.isbn.unique())
    return (train_users, train_items, test_users, test_items)


def country_to_usa_vs_world(data):
    features = []
    for ind in data.index:
        if data['country'][ind] == "Usa":
            features.append('Usa')
        else:
            features.append('rest')
    #df_users['usa_vs_rest'] = features
    return features

### Print functions ###

# Print Matrix Dimensions for training/test sets
def print_matrix_dim(x_set, set_name=""):
    print("Matrix Dimensions for ", set_name)
    print(set_name, " shape: {}".format(x_set.shape))
    print(set_name, " unique users: {}".format(x_set.user_id.nunique()))
    print(set_name, " unique items: {}".format(x_set.isbn.nunique()))
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
    unique_items = x_set.isbn.nunique()
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
    filename = OUTPUT_PATH + 'rec_scores.csv'
    with open(filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter='|', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(["Nr\t", "P@K\t\t\t", "R@K\t\t\t", "HR\t\t\t"])

    for round in all_results.keys():
        result = all_results[round]
        with open(filename, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow([round, result["p"], result["r"], result["hr"]])


def write_clf_scores_to_csv(all_results, metric):
    filename = OUTPUT_PATH + 'clf_' + metric + '_score.csv'
    with open(filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(
            ["Clf\t", "USA_vs_rest\t\t\t", ])

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
                [clf, output[clf]["usa_vs_rest"]])


def write_clf_preds_to_csv(result):
    # Make a separate file for each clf - attr pair
    base = OUTPUT_PATH +  "prediction_values/clf_pred"
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

    # Load user info
    df_users = pd.read_csv('data/bx-pre/users_top_' + BX_SIZE + '.csv', sep=',')
    df_users = df_users.reset_index(drop=True)
    print(df_users)

    pre_fix = 'co' if INCLUDE_FEATURES['country'] == True else 'usa'
    # Load user features
    user_features =  pd.read_csv('user_features/bx/feat_' + pre_fix +'_' + BX_SIZE + '.csv', sep=',')
    user_features = user_features.sort_values(by=['user_id'])
    user_features = user_features.reset_index(drop=True)
    print(user_features.shape)

    # Load interaction data and create training and test sets
    df_ratings = pd.read_csv('data/bx-pre/ratings_top_' + BX_SIZE + '.csv', sep=',')
    df_ratings = df_ratings.reset_index(drop=True)
    #print(df_ratings.dtypes)
    #df_ratings = df_ratings.drop(columns=["book_rating"], axis=1)
    users = np.sort(df_ratings.user_id.unique())

    # Create train and test sets
    (X_train1, X_test1) = prepare_rec_splits(df_ratings, train_size=0.4, test_size=0.3)
    (X_train2, X_test2) = prepare_rec_splits(df_ratings, train_size=0.7, test_size=0.3)
    

   
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
    #print("X train and test 1")
    #print_user_item_stats(train_users1, test_users1, "users")
    #print_user_item_stats(train_items1, test_items1, "items")
    #print("X train and test 2")
    #print_user_item_stats(train_users2, test_users2, "users")
    #print_user_item_stats(train_items2, test_items2, "items")

    #print(X_train1.dtypes)
    #print(user_features.dtypes)
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
    #write_double_rec_to_csv(recommendations_train, scores_train, recommendations_test, scores_test)
    write_rec_scores_to_csv(rec_scores)

 
    
    # Classifier
    # X = interaction_data + extra attributes
    # y = true value for attribute


    # Classification
    attributes_train = {}
    attributes_test = {}
    attributes ={}

    isbn_map = {}
    unique_isbn = df_ratings['isbn'].unique()
    print('unique_isbn: ', unique_isbn)
    print('unique_isbn len: ', len(unique_isbn))

    for isbn in unique_isbn:
        isbn_map[isbn] = len(isbn_map)

    rec_train = recs_to_matrix(recommendations_train, isbn_map)
    rec_test = recs_to_matrix(recommendations_test, isbn_map)
    print("Rec train shape: ", rec_train.shape)
    print("Rec test shape: ", rec_test.shape)

    results = []

    for attr in INFER_ATTR.keys():
        if INFER_ATTR[attr] == True:
            # Prepare gender attributes for classification
            #attributes_train[attr] = prepare_attributes_for_classifier(user_info, test_users1, attr_type=attr)
            #attributes_test[attr] = prepare_attributes_for_classifier(user_info, test_users2, attr_type=attr)
            #print("Diff in users: ", get_coldstart_units(test_users1, test_users1, unit_name="users"))
            # print(attributes_test[attr])
            df_train = df_users[df_users['user_id'].isin(test_users1)]
            df_test = df_users[df_users['user_id'].isin(test_users2)]
            

            attributes_train[attr] = country_to_usa_vs_world(df_train)
            attributes_test[attr] = country_to_usa_vs_world(df_test)
            attributes[attr] = country_to_usa_vs_world(df_users)

            print(attr)
            
            #all_attr = np.concatenate((attributes_train[attr], attributes_test[attr]), axis=0)
            #print(len(all_attr))
            counter_attr = collections.Counter(attributes[attr] )
            print("all_attr: ", counter_attr )
            
            print("all_attr unique: ", len(np.unique(attributes[attr])))
            print("less than 2 samples: ", count_attr_samples(counter_attr, 1, False) )
            print("2 samples: ", count_attr_samples(counter_attr, 2, False) )
            print("5 samples or less: ", count_attr_samples(counter_attr, 5, True) )
            #print("train_attr: ", collections.Counter(attributes_train[attr]))
            print("train_attr unique: ", len(np.unique(attributes_train[attr])))
            #print("test_attr: ", collections.Counter(attributes_test[attr]))
            print("test_attr unique: ", len(np.unique(attributes_test[attr])))

            print("train_attr: ", len(attributes_train[attr]))
            print("test_attr: ", len(attributes_test[attr]))
            print("\n\n")
            

    for attr in INFER_ATTR.keys():
        if INFER_ATTR[attr] == True:



            for clf in CLASSIFIERS.keys():
                if CLASSIFIERS[clf] == True:
                    print('\n')
                    print("CV with ", clf, " for ", attr)
                    result = {}
                    result['clf'] = clf
                    result['attr'] = attr
                    (result['acc'], result['f1'], result['auc']) = cross_validate(clf, rec_train, rec_test, attributes_train[attr], attributes_test[attr], attr, df_users)
                    results.append(result)
    write_clf_scores_to_csv(results, 'acc')
    write_clf_scores_to_csv(results, 'f1')
    write_clf_scores_to_csv(results, 'auc')



if __name__ == "__main__":
    main()
