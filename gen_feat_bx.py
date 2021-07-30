
import pandas as pd
from uszipcode import SearchEngine
import re
import numpy as np
import collections
from matplotlib import pyplot as plt

# Constants
"""
OCC_NUM = 21

STATE = "state"
CITY = "major_city"
COUNTY = "county"

LOC_TYPE = COUNTY
"""
AGE_GROUPS = {
    "age_0": [0,34],
    "age_1": [35,45],
    "age_2": [46,99],
    "age_3": [-1,-1]
}

USA_CAT = {
    "usa": 0,
    "rest": 1
}


BX_SIZE = '0_5'

BX_SIZES = ['0_1','0_3', '0_5', '0_7']


INCLUDE_FEATURES = {
    "age" : False,
    "city" : False,
    "state" : False,
    "country" : True,
    "usa" : False
    }



def load_bx_user_data():
    filename = "data/bx-pre/users_top_" + BX_SIZE + ".csv"
    df_users = pd.read_csv(filename, sep=',', encoding='ansi')

    if INCLUDE_FEATURES['usa'] == True:
        feat_usa = []
        feat_rest = []
        for ind in df_users.index:
            if df_users['country'][ind] == "Usa":
                feat_usa.append(1)
                feat_rest.append(0)
            else:
                feat_rest.append(1)
                feat_usa.append(0)
        df_users['usa'] = feat_usa
        df_users['rest'] = feat_rest

    elif INCLUDE_FEATURES['country'] == True:
        country_map = {}
        unique_countries = df_users['country'].unique()
        print('unique_countries: ', unique_countries)
        print('unique_countries len: ', len(unique_countries))
        for country in unique_countries:
            country_map[country] = len(country_map)

        mapped_countries=[]
        for ind in df_users.index:
            mapped_country = country_map[df_users['country'][ind]]
            mapped_countries.append(mapped_country)
        features = np.zeros((len(mapped_countries), len(unique_countries)), dtype=np.double)
        for ind in range(len(mapped_countries)):
            features[ind][mapped_countries[ind]] =1
        for ind in range(len(unique_countries)):
            df_users[ind] = features[:,ind]

    countries = df_users['country'].to_list()
    cnt = collections.Counter(countries)
    print(cnt)

    counter_values = []
    labels = []
    for key in cnt.keys():
        counter_values.append(cnt[key])
        if key == 'N/A, ':
            key='nan'
        labels.append(str(key))
    print(labels)

    plt.bar(labels, height=counter_values)
    plt.title('Countries', fontsize=14)
    plt.xlabel('attribute class')
    plt.ylabel('users')
    #plt.xticks(counter_values, labels)
    plt.show()
    df_users.drop(['country'], axis=1, inplace=True)
    df_users.drop(['age'], axis=1, inplace=True)

    #print("df_users: ", df_users)
    print("df_users shape: ", df_users.shape)
    
    return df_users


     


# Generate a name for the new user features file
def get_new_file_name():
    filename = "user_features/bx/feat"
    if INCLUDE_FEATURES["country"] == True:
        filename += "_co"
    if INCLUDE_FEATURES["usa"] == True:
        filename += "_usa"
    elif INCLUDE_FEATURES["state"] == True:
        filename += "_st"
    elif INCLUDE_FEATURES["city"] == True:
        filename += "_ci"
    filename += "_" + BX_SIZE + ".csv"
    return filename


def main():
    
    user_features = load_bx_user_data()
    filename = get_new_file_name()
    #user_features.to_csv(filename, index=False)
    print("User features saved to file: ", filename)

    

    
    """
    print("len user_info: ", len(user_info))
    print("len user_features: ", len(user_features))

    print("user_info[0]: ", user_info[1])
    print("user_features[0]: ", user_features[0])

    
    
    if INCLUDE_FEATURES["country"] == True:
        user_features = add_location_feature(user_info, user_features, "country")
    
    if INCLUDE_FEATURES["state"] == True:
        user_features = add_location_feature(user_info, user_features, "state")

    if INCLUDE_FEATURES["city"] == True:
        user_features = add_location_feature(user_info, user_features, "city")
    
    
    print("user_features[0]:", user_features[1])

    user_features = pd.DataFrame(data=user_features)
    
    print("user_features shape: ", user_features.shape)

    filename = get_new_file_name()
    user_features.to_csv(filename, index=False)
    print("User features saved to file: ", filename)"""







"""
# load other user data -> age, gender ...
def load_user_data(filename="bx-users.csv", path="data/book-crossing/"):
    
    user_info = {}
    user_features = []
    with open(path+filename, 'r') as fin:
        for line in fin.readlines():
            attributes = line.split(";")
            if len(attributes) == 3:
                user_id, locations, age = line.split(";")   
                
                # Preparing location
                locs = locations.split(", ")
                if len(locs) != 3:
                    city = None
                    state = None
                    country = None
                else: 
                    (city, state, country) = locations.split(', ')

                # Preparing age
                age = re.sub(r'[^a-zA-Z0-9]', '',age)
                if age.isnumeric():
                    age = int(age)
                else:
                    age = -1

                # Preparing user_id
                user_id = re.sub(r'[^a-zA-Z0-9]', '',user_id)

                # Preparing user_info
                user_info[int(user_id)-1] = {
                    'city' : city,
                    'region' : state,
                    'country' : country,
                    'age': age
                }
                user_features.append({"user_id": str(user_id)})
        print('User Info Loaded!')
    return (user_info, user_features)
    """


if __name__ == "__main__":
    main()