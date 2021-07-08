
import pandas as pd
from uszipcode import SearchEngine
import re
import numpy as np

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



INCLUDE_FEATURES = {
    "age" : False,
    "city" : False,
    "state" : False,
    "country" : True
    }



def load_bx_user_data(filename="users_top.csv", path="data/bx-pre/"):
    df_users = pd.read_csv(path + filename, sep=',', encoding='ansi')
    user_features = []
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
    df_users.drop(['country'], axis=1, inplace=True)
    df_users.drop(['age'], axis=1, inplace=True)

    print("df_users: ", df_users)
    print("df_users shape: ", df_users.shape)
    return df_users


        

# load other user data -> age, gender ...
def load_user_data(filename="users_top.csv", path="data/bx-pre/"):
    user_mapping = {}
    user_info = {}
    user_features = []
    users_tmp = []
    with open(path+filename, 'r') as fin:
        for line in fin.readlines():
            attributes = line.split(",")
            user_id,  age, country, = line.split(",")   
            users_tmp.append(str(user_id))
    users_tmp = np.unique(users_tmp)
    for ind in range(len(users_tmp)):
        user_mapping[str(users_tmp[ind])] = ind
    
    with open(path+filename, 'r') as fin:
        for line in fin.readlines():
            attributes = line.split(",")
            user_id,  age, country, = line.split(",")   
            
            if age.isnumeric() == False:
                age = -1.0
            if country == '' or ',' in country:
                country = 'nan'
            user_ind = user_mapping[user_id]
            user_info[user_ind] = {
                'age': int(age),
                'country': country
            }
            user_info.append({"user_id": user_ind})
        
    print('User Info Loaded!')
    return (user_info, user_features, user_mapping)


# Add age to user features
def add_age_feature(user_info, user_features):

    def is_in_age_group(age, age_cat):
        return 1 if age >= AGE_GROUPS[age_cat][0] and age <= AGE_GROUPS[age_cat][1] else 0
    
    for i in range(len(user_info)):
        for j in range(len(AGE_GROUPS)):
            label = "age_" + str(j)
            age = user_info[i]["age"]
            user_features[i][label] = is_in_age_group(age, label)
        
    print("Age feature was added")
    return user_features


# Add location to user features
def add_location_feature(user_info, user_features, loc_type):
    # Generate locs dict
    locs = {}
    for u_id in user_info.keys():
        loc = user_info[u_id][loc_type]

        if loc not in locs.keys():
            locs[loc] = loc_type + "_" + str(len(locs))
    print("locs: ", locs)

    for i in range(len(user_features)):
        for l in locs:
            loc_label = locs[l]
            user_features[i][loc_label] = 0

    # Generate location features
    for i in range(len(user_features)):
        u_id = user_features[i]['user_id']
        user_loc = user_info[u_id][loc_type]

        loc_label = locs[user_loc]
        user_features[i][loc_label] = 1
    
    print(loc_type ," feature was added")
    return user_features



# Generate a name for the new user features file
def get_new_file_name():
    filename = "user_features/bx/feat"
    if INCLUDE_FEATURES["country"] == True:
        filename += "_co"
    elif INCLUDE_FEATURES["state"] == True:
        filename += "_st"
    elif INCLUDE_FEATURES["city"] == True:
        filename += "_ci"
    filename += ".csv"
    return filename


def main():
    
    user_features = load_bx_user_data()
    filename = get_new_file_name()
    user_features.to_csv(filename, index=False)
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