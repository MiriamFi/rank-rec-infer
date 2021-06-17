from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import OneHotEncoder
import pandas as pd


y1 = [0, 1, 0, 1]
y2 = [0, 1, 2, 3, 4, 5, 6, 7, 8]
y1 = pd.DataFrame(data=y1)
y2 = pd.DataFrame(data=y2)

enc = OneHotEncoder()
y_enc1 = enc.fit_transform(y1).toarray()
y_enc2 = enc.fit_transform(y2).toarray()
y_bin1= label_binarize(y1, classes=[0,1])
y_bin2= label_binarize(y2, classes=[0,1,2,3, 4, 5, 6, 7, 8])

print("y_enc1: ", y_enc1)
print("y_enc2: ", y_enc2)
print("y_bin1: ", y_bin1)
print("y_bin2: ", y_bin2)