import pandas as pd
import numpy as np
import copy

import re
from string import ascii_letters, digits

#TODO: Switch user filtering and book filtering
#TODO: Turn everything into implicit ratings
#TODO: Figure out how to preprocess the location info


path = 'data/book-crossing/'

df_ratings = pd.read_csv(path + 'bx-ratings.csv', sep=';', encoding='ansi')
df_books = pd.read_csv(path + 'bx-books.csv', sep=';', encoding='ansi', escapechar='\\')
df_users = pd.read_csv(path + 'bx-users.csv', sep=';', encoding='ansi')

def colname_fix(colname):
    colname = colname.lower()
    return re.sub( "-", "_", colname)

def ascii_check_bulk(df):
    for col in df.columns:
        print('items with non-ascii characters in %s: %d' % (col, df[col].apply(ascii_check).sum()))
    print('')

def ascii_check(item):
    for letter in str(item):
        if letter not in ascii_letters + digits:
            return 1
        else:
            return 0

for df in [df_ratings, df_books, df_users]:
    df.columns = [ colname_fix(col) for col in df.columns]
    print(df.columns)


print('Ratings:\nNumber of ratings: %d\nNumber of books: %d\nNumber of users: %d' % (len(df_ratings), len(df_ratings['isbn'].unique()), len(df_ratings['user_id'].unique())))
print('\nNumber of books: %d' % len(df_books))
print('\nNumber of users: %d' % len(df_users))

ascii_check_bulk(df_ratings)
ascii_check_bulk(df_books)
ascii_check_bulk(df_users)



#### Filtering observations
#Remove (incorrect) ISBN with non-ascii characters
#Use only country instead of whole 'location' data
#Remove images' urls
#Separate explicit (1-10) and implicit (0) ratings


df_ratings['isbn_check'] = df_ratings['isbn'].apply(ascii_check)
df_ratings = df_ratings[df_ratings['isbn_check']==0]

df_users['country'] = df_users['location'].apply(lambda x: x.split(', ')[-1].title())
df_users['country_check'] = df_users['country'].apply(ascii_check)
df_users.loc[df_users['country_check']==1, 'country'] = np.nan

US_mappings = ["Indiana,", "Illinois,", "New Hampshire,", "Louisiana,", "Texas,", "Oregon,",
 "Massachusetts,", "Virginia,",  "California,",  "North Carolina,",  "South Carolina,",
  "Auckland,", "United States", "Kentucky,", "Pennsylvania,", "Missouri,", "New Jersey,", 
  "New York,", "Tennessee,", "Oklahoma,", "United State", "Mississippi,", "Us", "Michigan," ]
UK_mappings = ['England']
Italy_mappings = ["Piemonte,"]

countries = []
for ind in df_users.index:
    country = df_users['country'][ind]
    #print('country: ', country)
    if country in US_mappings:
        country = 'Usa'
    elif country in UK_mappings:
        country = 'United Kingdom'
    elif country in Italy_mappings:
        country = 'Italy'
    elif country in Italy_mappings:
       country = 'Italy'
    elif country == 'N/A' or country== 'Far Away...':
        country = np.nan
    countries.append(country)
        
df_users_m = copy.deepcopy(df_users)
df_users_m.drop(columns=['country'], axis=1)
df_users_m['country'] = countries
    



#df_users['state'] = df_users['location'].apply(lambda x: x.split(', ')[1].title())
#df_users['state_check'] = df_users['state'].apply(ascii_check)
#df_users.loc[df_users['state_check']==1, 'state'] = np.nan

#df_users['city'] = df_users['location'].apply(lambda x: x.split(', ')[0].title())
#df_users['city_check'] = df_users['city'].apply(ascii_check)
#df_users.loc[df_users['city_check']==1, 'city'] = np.nan

df_ratings.drop(['isbn_check'], axis=1, inplace=True)
df_books.drop(['image_url_s', 'image_url_m', 'image_url_l'], axis=1, inplace=True)
df_users_m.drop(['country_check'], axis=1, inplace=True)
df_users_m.drop(['location'], axis=1, inplace=True)
#df_users.drop(['state_check'], axis=1, inplace=True)
#df_users.drop(['city_check'], axis=1, inplace=True)

df_ratings_explicit = df_ratings[df_ratings['book_rating']!=0]
df_ratings_implicit = df_ratings[df_ratings['book_rating']==0]

print('Explicit ratings: %d\nImplicit ratings: %d' % (len(df_ratings_explicit), len(df_ratings_implicit)))


#df_ratings_explicit.to_csv('data/ratings_explicit.csv', encoding='utf-8', index=False)
#df_ratings_implicit.to_csv('data/ratings_implicit.csv', encoding='utf-8', index=False)
#df_books.to_csv('data/books.csv', encoding='utf-8', index=False)
#df_users.to_csv('data/users.csv', encoding='utf-8', index=False)


## Reduce dimensionality
# Users with at least 20 ratings and top 20% most frequently rated books




book_ratings_threshold_perc = 0.5
book_ratings_threshold = len(df_ratings_implicit['isbn'].unique()) * book_ratings_threshold_perc

filter_books_list = df_ratings_implicit['isbn'].value_counts().head(int(book_ratings_threshold)).index.to_list()
df_ratings_top =df_ratings_implicit[df_ratings_implicit['isbn'].isin(filter_books_list)]

print('Filter: top %d%% most frequently rated books\nNumber of records: %d' % (book_ratings_threshold_perc*100, len(df_ratings_top)))


user_ratings_threshold = 20

filter_users = df_ratings_top['user_id'].value_counts()
filter_users_list = filter_users[filter_users >= user_ratings_threshold].index.to_list()

df_ratings_top_new = df_ratings_top[df_ratings_top['user_id'].isin(filter_users_list)]

print('Filter: users with at least %d ratings\nNumber of records: %d' % (user_ratings_threshold, len(df_ratings_top_new)))
print('Filter: unique users with at least %d ratings\nNumber of records: %d' % (user_ratings_threshold, len(df_ratings_top_new['user_id'].unique())))

df_ratings_top_new.to_csv('data/bx-pre/ratings_top.csv', encoding='utf-8', index=False)



df_top_users = df_users_m[df_users_m['user_id'].isin(filter_users_list)]

print('Filter: users with at least %d ratings\nNumber of records: %d' % (user_ratings_threshold, len(df_top_users)))

df_top_users.to_csv('data/bx-pre/users_top.csv', encoding='utf-8', index=False)



"""
user_ratings_threshold = 20

filter_users = df_ratings_explicit['user_id'].value_counts()
filter_users_list = filter_users[filter_users >= user_ratings_threshold].index.to_list()

df_ratings_top = df_ratings_explicit[df_ratings_explicit['user_id'].isin(filter_users_list)]

print('Filter: users with at least %d ratings\nNumber of records: %d' % (user_ratings_threshold, len(df_ratings_top)))


book_ratings_threshold_perc = 0.5
book_ratings_threshold = len(df_ratings_top['isbn'].unique()) * book_ratings_threshold_perc

filter_books_list = df_ratings_top['isbn'].value_counts().head(int(book_ratings_threshold)).index.to_list()
df_ratings_top = df_ratings_top[df_ratings_top['isbn'].isin(filter_books_list)]

print('Filter: top %d%% most frequently rated books\nNumber of records: %d' % (book_ratings_threshold_perc*100, len(df_ratings_top)))"""