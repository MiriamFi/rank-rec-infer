
import pandas as pd

# Variables
OCC_NUM = 21
AGE_GROUPS = {
    "age_0": [0,34],
    "age_1": [35,45],
    "age_2": [46,99]
}


include_features = {
        "gender" : False,
        "age" : False,
        "occupation" : False,
        "location" : False
        }
    

# load other user data -> age, gender ...

def load_user_data(filename="u.user", path="ml-100k/"):
    user_info = {}
    user_features = []
    with open(path+filename, 'r') as fin:
        for line in fin.readlines():
            user_id, age, gender, occu, zipcode = line.split('|')
            user_info[int(user_id)-1] = {
                'age': int(age),
                'gender': gender,
                'occupation': occu,
                'zipcode': zipcode
            }
            user_features.append({"user_id": str(user_id)})
        print('User Info Loaded!')
    return user_info, user_features




def add_gender_feature(user_info, user_features):
    if include_features["gender"] == True:
        return

    for i in range(len(user_info)):
        user_features[i]["gender_F"] = 1 if user_info[i]["gender"] == "F" else 0
        user_features[i]["gender_M"] = 1 if user_info[i]["gender"] == "M" else 0
    include_features["gender"] = True
    print("Gender feature was added")
    return user_features

def add_age_feature(user_info, user_features):
    if include_features["age"] == True:
        return
    
    def is_in_age_group(age, age_cat):
        return 1 if age >= AGE_GROUPS[age_cat][0] and age <= AGE_GROUPS[age_cat][1] else 0
    
    for i in range(len(user_info)):
        for j in range(len(AGE_GROUPS)):
            label = "age_" + str(j)
            age = user_info[i]["age"]
            user_features[i][label] = is_in_age_group(age, label)

    include_features["age"] = True
    print("Age feature was added")
    return user_features

def add_occupation_feature(user_info, user_features):
    if include_features["occupation"] == True:
        return
    occupations = {}
    for u_id in range(len(user_info)):
        # Generate all occupation feature labels
        for o in range(OCC_NUM):
            feature_label = "occupation_" + str(o)
            user_features[u_id][feature_label] = 0
        
        # Add occupation feature
        occ = user_info[u_id]["occupation"]

        if occ not in occupations.keys():
            occupations[occ] = "occupation_" + str(len(occupations))

        feature_label = occupations[occ]
        user_features[u_id][feature_label] = 1
    
    include_features["occupation"] = True
    print("Occupation feature was added")
    return user_features





def get_new_file_name():
    filename = "user_features/feat"
    if include_features["gender"] == True:
        filename += "_g"
    if include_features["age"] == True:
        filename += "_a"
    if include_features["occupation"] == True:
        filename += "_o"
    if include_features["location"] == True:
        filename += "_l"
    filename += ".csv"
    return filename

def main():
    
    user_info, user_features = load_user_data()
    
    print("len user_info: ", len(user_info))
    print("len user_features: ", len(user_features))

    user_features = add_gender_feature(user_info, user_features)
    
    print("user_features[0]:", user_features[0])

    user_features = add_age_feature(user_info, user_features)
    

    user_features = add_occupation_feature(user_info, user_features)
    
    print("user_features[0]:", user_features[0])

    user_features = pd.DataFrame(data=user_features)
    
    print("user_features shape: ", user_features.shape)

    filename = get_new_file_name()
    user_features.to_csv(filename, index=False)
    print("User features saved to file: ", filename)





if __name__ == "__main__":
    main()