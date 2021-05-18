from uszipcode import SearchEngine
STATE = "state"
CITY = "major_city"

LOC_TYPE = STATE

# Shold work
search = SearchEngine(simple_zipcode=True)
zipcode = search.by_zipcode("85711")
zipcode = zipcode.to_dict()
print(zipcode)
print("state: ", zipcode[LOC_TYPE])
print("city: ", zipcode[CITY])


# Error test 1
zipcode = search.by_zipcode("66315")
zipcode = zipcode.to_dict()
print("\nzip: ", zipcode)
print("state: ", zipcode[LOC_TYPE])
print("city: ", zipcode[CITY])


# Error test 2
zipcode = search.by_zipcode("41850")
zipcode = zipcode.to_dict()
print("\nzip: ", zipcode)
print("state: ", zipcode[LOC_TYPE])
print("city: ", zipcode[CITY])

# Error test 3
zipcode = search.by_zipcode("T8H1N")
zipcode = zipcode.to_dict()
print("\nzip: ", zipcode)
print("state: ", zipcode[LOC_TYPE])
print("city: ", zipcode[CITY])


# Error test 4
zipcode = search.by_zipcode("46005")
zipcode = zipcode.to_dict()
print("\nzip: ", zipcode)
print("state: ", zipcode[LOC_TYPE])
print("city: ", zipcode[CITY])


# Error test 5
zipcode = search.by_zipcode("V3N4P")
zipcode = zipcode.to_dict()
print("\nzip: ", zipcode)
print("state: ", zipcode[LOC_TYPE])
print("city: ", zipcode[CITY])


# Error test 6
zipcode = search.by_zipcode("L9G2B")
zipcode = zipcode.to_dict()
print("\nzip: ", zipcode)
print("state: ", zipcode[LOC_TYPE])
print("city: ", zipcode[CITY])


# Error test 7
zipcode = search.by_zipcode("00000")
zipcode = zipcode.to_dict()
print("\nzip: ", zipcode)
print("state: ", zipcode[LOC_TYPE])
print("city: ", zipcode[CITY])