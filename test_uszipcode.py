from uszipcode import SearchEngine
STATE = "state"
CITY = "major_city"

LOC_TYPE = STATE

search = SearchEngine(simple_zipcode=True)
zipcode = search.by_zipcode("85711")
zipcode = zipcode.to_dict()
print(zipcode)
print("state: ", zipcode[LOC_TYPE])
print("city: ", zipcode[CITY])

zipcode = search.by_zipcode("02138")
zipcode = zipcode.to_dict()
print(zipcode)
print("state: ", zipcode[LOC_TYPE])
print("city: ", zipcode[CITY])