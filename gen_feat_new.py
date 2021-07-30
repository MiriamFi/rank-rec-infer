
import pandas as pd
from uszipcode import SearchEngine
import collections

# Constants
OCC_NUM = 21

STATE = "state"
CITY = "major_city"
COUNTY = "county"
AREA_5 = "area_5"
AREA_2 = "area_2"
COAST = "coast"


LOC_TYPE = COAST

AGE_GROUPS = {
    "age_0": [0,34],
    "age_1": [35,45],
    "age_2": [46,99]
}
INCLUDE_FEATURES = {
        "gender" : True,
        "age" : True,
        "occupation" : True,
        "location" : True
        }

STATE_AREA_5 = {
    'west' : ['WA', 'OR', 'ID', 'MT', 'WY', 'CO', 'UT', 'NV', 'CA', 'AK', 'HI'],
    'midwest' : ['ND', 'SD', 'NE', 'KS', 'MN', 'IA', 'MO', 'WI', 'IL', 'IN', 'MI', 'OH'],
    'southwest' : ['AZ', 'NM', 'OK', 'TX'],
    'northeast' : ['NY', 'PA', 'NJ', 'CT', 'RI', 'MA', 'NH', 'ME', 'VT'],
    'southeast' : ['AR', 'LA', 'MS', 'AL', 'GA', 'FL', 'SC', 'NC', 'VA', 'DC', 'DE','MD', 'WV', 'KY','TN'],
    'none' : None
}

STATE_COAST = {
    'coast' : ['WA','OR','CA','AK','HI', 'TX', 'LA', 'MS', 'AL', 'GA', 'FL', 'SC', 'NC', 'VA', 'DC', 'DE','MD','NY', 'PA', 'NJ', 'CT', 'RI', 'MA', 'NH', 'ME', 'VT','MN','WI','IL', 'IN', 'MI', 'OH'],
    'rest': ['ID', 'MT', 'WY', 'CO', 'UT', 'NV', 'ND', 'SD', 'NE', 'KS',  'IA', 'MO', 'AZ', 'NM', 'OK', 'AR', 'WV', 'KY','TN'],
    'none' : None

}

STATE_AREA_2 = {
    'west' : ['ND', 'SD', 'NE', 'KS', 'WA', 'OR', 'ID', 'MT', 'WY', 'CO', 'UT', 'NV', 'CA', 'AK', 'HI', 'AZ', 'NM', 'OK', 'TX'],
    'east' : ['MN', 'IA', 'MO', 'WI', 'IL', 'IN', 'MI', 'OH', 'NY', 'PA', 'NJ', 'CT', 'RI', 'MA', 'NH', 'ME', 'VT', 'AR', 'LA', 'MS', 'AL', 'GA', 'FL', 'SC', 'NC', 'VA', 'DC', 'DE','MD', 'WV', 'KY','TN'],
    'none' : None
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


# Variables
added_features = {
        "gender" : False,
        "age" : False,
        "occupation" : False,
        "location" : False
        }


# load other user data -> age, gender ...
def load_user_data(filename="u.user", path="data/ml-100k/"):
    user_info = {}
    user_features = []
    with open(path+filename, 'r') as fin:
        for line in fin.readlines():
            user_id, age, gender, occu, zipcode = line.split('|')
            user_info[int(user_id)-1] = {
                'age': int(age),
                'gender': gender,
                'occupation': occu,
                'zipcode': str(zipcode)
            }
            user_features.append({"user_id": str(user_id)})
        print('User Info Loaded!')
    return user_info, user_features


# Add gender to user features
def add_gender_feature(user_info, user_features):
    if added_features["gender"] == True:
        return

    for i in range(len(user_info)):
        user_features[i]["gender_F"] = 1 if user_info[i]["gender"] == "F" else 0
        user_features[i]["gender_M"] = 1 if user_info[i]["gender"] == "M" else 0
    added_features["gender"] = True
    print("Gender feature was added")
    return user_features

# Add age to user features
def add_age_feature(user_info, user_features):
    if added_features["age"] == True:
        return
    
    def is_in_age_group(age, age_cat):
        return 1 if age >= AGE_GROUPS[age_cat][0] and age <= AGE_GROUPS[age_cat][1] else 0
    
    for i in range(len(user_info)):
        for j in range(len(AGE_GROUPS)):
            label = "age_" + str(j)
            age = user_info[i]["age"]
            user_features[i][label] = is_in_age_group(age, label)
        
    added_features["age"] = True
    print("Age feature was added")
    return user_features

# Add occupation to user features
def add_occupation_feature(user_info, user_features):
    
    if added_features["occupation"] == True:
        return
    
    # Generate occupations dict
    occupations = {}
    for u_id in range(len(user_info)):
        occ = user_info[u_id]["occupation"]
        occ = OCCUPATIONS[occ]
        #if occ == 'none':
            #occ = 'other'

        if occ not in occupations.keys():
            occupations[occ] = "occupation_" + str(len(occupations))
    print("Occupations: ", occupations)

    # Generate occupation features
    for u_id in range(len(user_info)):

        for o in occupations:
            occ_label = occupations[o]
            user_features[u_id][occ_label] = 0
        
        user_occ = user_info[u_id]["occupation"]
        #if user_occ == 'none':
            #user_occ = 'other'
        user_occ = OCCUPATIONS[occ]

        occ_label = occupations[user_occ]
        user_features[u_id][occ_label] = 1
    
    added_features["occupation"] = True
    print("Occupation feature was added")
    return user_features

# Add location to user features
def add_location_feature(user_info, user_features):
    if LOC_TYPE== AREA_5:
        state_area = STATE_AREA_5
    elif LOC_TYPE== AREA_2:
        state_area = STATE_AREA_2
    elif LOC_TYPE== COAST:
        state_area = STATE_COAST
    
    if added_features["location"] == True:
        return
    
    locations = []
    
    # Map zip code to state/city
    def map_location(zipcode):
        search = SearchEngine(simple_zipcode=True)
        zip_code = search.by_zipcode(zipcode)
        zip_code = zip_code.to_dict()
        area = map_state_to_area(zip_code['state'])
        return area
    
    def map_state_to_area(state):
        for key in state_area.keys():
            if state == None:
                locations.append('none')
                return 'none'
            elif state in state_area[key]:
                locations.append(key)
                return key
        return None

    
    # Generate user features
    for u_id in range(len(user_info)):
        for loc_key in state_area.keys():
            user_features[u_id][loc_key] = 0
        
        zip_code = user_info[u_id]["zipcode"].strip()
        location = map_location(zip_code)

        user_features[u_id][location] = 1
    
    added_features["location"] = True
    print("Location feature was added")
    cnt = collections.Counter(locations)
    print(cnt)
    print("Len unique locations: ", len(cnt))
    return user_features



# Generate a name for the new user features file
def get_new_file_name():
    filename = "user_features/ml_new/feat"
    if added_features["gender"] == True:
        filename += "_g"
    if added_features["age"] == True:
        filename += "_a"
    if added_features["occupation"] == True:
        filename += "_o"
    if added_features["location"] == True:
        if LOC_TYPE == AREA_5:
            filename += "_r5"
        if LOC_TYPE == AREA_2:
            filename += "_r2"
        if LOC_TYPE == COAST:
            filename += "_coast"
        
    filename += ".csv"
    return filename


def main():
    
    user_info, user_features = load_user_data()
    
    print("len user_info: ", len(user_info))
    print("len user_features: ", len(user_features))

    if INCLUDE_FEATURES["gender"] == True:
        user_features = add_gender_feature(user_info, user_features)
    
    if INCLUDE_FEATURES["age"] == True:
        user_features = add_age_feature(user_info, user_features)
    
    if INCLUDE_FEATURES["occupation"] == True:
        user_features = add_occupation_feature(user_info, user_features)

    if INCLUDE_FEATURES["location"] == True:
        user_features = add_location_feature(user_info, user_features)
    
    
    print("user_features[0]:", user_features[0])

    user_features = pd.DataFrame(data=user_features)
    
    print("user_features shape: ", user_features.shape)

    filename = get_new_file_name()
    user_features.to_csv(filename, index=False)
    print("User features saved to file: ", filename)





if __name__ == "__main__":
    main()