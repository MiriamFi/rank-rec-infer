from uszipcode import SearchEngine
import us
STATE = "state"
CITY = "major_city"

LOC_TYPE = STATE

# Shold work
search = SearchEngine(simple_zipcode=True)
zipcode = search.by_zipcode("8571100000")
zipcode = zipcode.to_dict()
print(zipcode)
print("state: ", zipcode[LOC_TYPE])
print("city: ", zipcode[CITY])




states_abbr = us.states.mapping('fips', 'abbr')
us_states = []
for k in states_abbr.keys():
    us_states.append(states_abbr[k])

unique_us_states = set(us_states)
print("us states: ", us_states)
print("unique us states: ", len(set(us_states)))
states = []
cnts = []


for key in states_abbr.keys():
    state = search.by_state(states_abbr[key])
    #print(state)
    cnt = 0
    for v in range(len(state)):
        zipcode = state[v].to_dict()
        if zipcode["state"] == None:
            cnt += 1
    cnts.append(cnt)
    states.append(zipcode["state"])

unique_states = set(states)

diff_states = unique_us_states - unique_states

print("States: ", states)
print("unique states: ", unique_states)
print("Len states: ", len(states))
print("Len unique states: ", len(unique_states))
print("cnts: ", cnts)
print("diff states: ", diff_states)



STATE_AREA = {
    'west' : ['WA', 'OR', 'ID', 'MT', 'WY', 'CO', 'UT', 'NV', 'CA', 'AK', 'HI'],
    'midwest' : ['ND', 'SD', 'NE', 'KS', 'MN', 'IA', 'MO', 'WI', 'IL', 'IN', 'MI', 'OH'],
    'southwest' : ['AZ', 'NM', 'OK', 'TX'],
    'northwest' : ['NY', 'PA', 'NJ', 'CT', 'RI', 'MA', 'NH', 'ME', 'VT'],
    'souheast' : ['AR', 'LA', 'MS', 'AL', 'GA', 'FL', 'SC', 'NC', 'VA', 'DC', 'DE','MD', 'WV', 'KY','TN']
}

area_dist = {}
total_states = 0
my_states = []
for key in STATE_AREA.keys():
    dist = len(STATE_AREA[key])
    area_dist[key] = dist
    total_states += dist
    for i in range(dist):
        my_states.append(STATE_AREA[key][i])

print("area distribution:", area_dist)
print("num of states: ", total_states)
unique_my_states = set(my_states) 
print("missing states: ", unique_states - unique_my_states)


new_states = []
cnts2 = []
for st in my_states:
    state = search.by_state(st)
    #print(state)
    cnt = 0
    for v in range(len(state)):
        zipcode = state[v].to_dict()
        if zipcode["state"] == None:
            cnt += 1
    cnts2.append(cnt)
    new_states.append(zipcode["state"])

unique_new_states = set(new_states)

diff_new_states = unique_states - unique_new_states

print("New states: ", new_states)
print("unique new states: ", unique_new_states)
print("Len new states: ", len(new_states))
print("Len unique new states: ", len(unique_new_states))
print("cnts2: ", cnts2)
print("diff new states: ", diff_new_states)