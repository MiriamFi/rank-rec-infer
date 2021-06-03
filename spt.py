
import numpy as np
import pandas as pd



# Load interaction data
def load_data(filename="u.data", path="ml-100k/"):
    data = [] # user id + movie id
    with open(path+filename) as f:
        for line in f:
            (user, movieid, rating, ts) = line.split('\t')
            
            if float(rating) >= 4.0:
                    r = 1.0
            else:
                r = 0.0

            data.append({ "user_id": int(user), "item_id": int(movieid), "rating": r, "ts": int(ts.strip())})


    # Prepare data
    data = pd.DataFrame(data=data)
    return data

def prepare_splits(data, train_size=0.9, test_size=0.1):
    ranks = data.groupby('user_id')['ts'].rank(method='first')
    counts = data['user_id'].map(data.groupby('user_id')['ts'].apply(len))
    thrs_train = (ranks/counts) <= train_size
    thres_train = pd.DataFrame(thrs_train, columns=["thrs_train"])
    data = data.join(thres_train)
    X_train = data[data['thrs_train'] == True]
    print("X_train: ", X_train)
    X_train = X_train.drop(columns='thrs_train', axis=1)
    print("X_train: ", X_train)
    if train_size + test_size == 1.0:
        X_test = data[data['thrs_train'] == False]
        print("X_test: ", X_test)
        X_test = X_test.drop(columns='thrs_train', axis=1)
        print("X_test: ", X_test)
    else:
        thrs_test = (ranks/counts) <= (train_size + test_size)
        thres_test = pd.DataFrame(thrs_test, columns=["thrs_test"])
        data = data.join(thres_test)
        data["thrs"] = data["thrs_test"] > data["thrs_train"]
        X_test = data[data['thrs'] == True]
        print("X_test: ", X_test)
        X_test = X_test.drop(columns=['thrs', 'thrs_train', 'thrs_test'], axis=1)
        print("X_test: ", X_test)
    
    
    
    


    return (X_train, X_test)



def main():
    data = load_data()
    print(data)
    prepare_splits(data)


if __name__ == "__main__":
    main()