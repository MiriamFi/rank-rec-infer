
import pandas as pd

# Variables


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
    for i in range(len(user_info)):
        user_features[i]["gender_F"] = 1 if user_info[i]["gender"] == "F" else 0
        user_features[i]["gender_M"] = 1 if user_info[i]["gender"] == "M" else 0
    include_features["gender"] = True
    print("Gender feature was added")
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

    user_features = pd.DataFrame(data=user_features)
    
    print("user_features shape: ", user_features.shape)

    filename = get_new_file_name()
    user_features.to_csv(filename, index=False)
    print("User features saved to file: ", filename)





if __name__ == "__main__":
    main()