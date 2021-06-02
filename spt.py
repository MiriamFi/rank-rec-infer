
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

def prepare_splits(data):
    ranks = data.groupby('user_id')['ts'].rank(method='first')
    counts = data['user_id'].map(data.groupby('user_id')['ts'].apply(len))
    thresholds = (ranks/counts) > 0.8
    thresholds = pd.DataFrame(thresholds, columns=["thrs"])
    print(thresholds)
    data = data.join(thresholds)
    print(data)
    X_train = data[data['thrs'] == False]
    X_test = data[data['thrs'] == True]
    print("X_train: ", X_train)
    print("X_test: ", X_test)
    return (X_train, X_test)



def main():
    data = load_data()
    print(data)
    prepare_splits(data)


if __name__ == "__main__":
    main()