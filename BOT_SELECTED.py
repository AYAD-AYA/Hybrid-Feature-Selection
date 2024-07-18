import glob
import os
from collections import Counter
import seaborn as sns
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, RandomOverSampler, SMOTENC
from keras import Sequential
from keras.layers import Dense
from matplotlib import pyplot as plt
from sklearn import preprocessing, metrics, svm
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import ConfusionMatrixDisplay, auc, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import time
from features_selection import proposed_fs_model



# Merged 4 files of the dataset into one named "merged" and ran it one time in any file and then commented it
# path = "E:\Datasets\iotbotnet\Entiredataset"
# allfiles = glob.glob(os.path.join(path, "*.csv"))
# all_csv = (pd.read_csv(f, sep=",") for f in allfiles)
# df_merged = pd.concat(all_csv, ignore_index=True)
# df_merged.to_csv("merged.csv")
#
file = pd.read_csv("../merged.csv")
file.drop(file.columns[0],axis=1,inplace=True)
# Create category+subcategory feature in one feature called Category
file['Category'] = file['category'] + file['subcategory']
file.drop('category',axis=1,inplace=True)
file.drop('subcategory',axis=1,inplace=True)
# # to uniform datatypes of this column
file['dport'] = file['dport'].astype('str')
file['sport'] = file['sport'].astype('str')
file.drop('pkSeqID',axis=1,inplace=True)
# drop this outlier
file.drop(file[file['Category'] =="TheftData_Exfiltration"].index, inplace = True)
print(file.columns.tolist())
#
all_features=file.columns
# Label encoder
encoder = LabelEncoder()
for column in all_features:
    file[column] = encoder.fit_transform(file[column])
print("all")

Category=file['Category']
print(Category)
selected_features = proposed_fs_model(file,'attack',10)
if 'Category' not in selected_features:
    selected_features.append('Category')
for i in file:
    if i in selected_features:
        pass
    else:
        file.drop(i, axis=1, inplace=True)
print("f")
print(file.columns)


# I insert this to a specific task in the second level, that I won't take the row detected
# as an attack with this seq, and I want to it with integer number and only in the first column
file.insert(0,'seq.ID',file.index)
print("seq")

x, y = file.drop('attack',axis=1),file['attack']
first_column = x.iloc[:, 0]
data_to_normalize = x.iloc[:, 1:]
scaler = preprocessing.MinMaxScaler()
normalized_data = scaler.fit_transform(data_to_normalize)
x = np.column_stack((first_column, normalized_data))
print("done20")
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.30, random_state=0)
print(y_train)
# oversampling
over_s = SMOTE()
x_train, y_train = over_s.fit_resample(x_train, y_train)
print("don90")
# model = DecisionTreeClassifier()
# model = RandomForestClassifier(n_estimators=10,criterion="entropy")
# model = KNeighborsClassifier(n_neighbors=5)
model = GaussianNB()

model.fit(x_train,y_train)
start = time.time()
y_pred = model.predict(x_test)
end = time.time()
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("precision:", metrics.precision_score(y_test, y_pred))
print("recall:", metrics.recall_score(y_test, y_pred))
print("f1score:", metrics.f1_score(y_test, y_pred))
cf=metrics.confusion_matrix(y_test, y_pred)
print(cf)
print(metrics.classification_report(y_test, y_pred))
print("predict time= ", end - start)
fmt = 'd'
sns.heatmap(cf, annot=True, cmap="Blues",fmt=fmt)
plt.xlabel("Predicted labels")
plt.ylabel("True labels")
plt.title("Confusion Matrix")
plt.show()

TP, FN = cf[0]
FP, TN = cf[1]
FPR = FP / (FP + TN)
FNR = FN / (FN + TP)
print(FPR,FNR)






# # LEVEL-2
y_pred = np.ravel(y_pred)
level2_data = x_test[y_pred==1]
level2_x=pd.DataFrame(level2_data)
indexx=level2_x.iloc[:, 0]
yy=[]
for index in indexx:
    yyy= file._get_value(index, 'Category')
    yy.append(yyy)

print("leny",len(yy))
print("y",np.unique(yy))
print("x",level2_x.shape)


x_train, x_test, y_train, y_test = train_test_split(level2_x, yy, test_size=.40, random_state=0,shuffle=True)



print(len(x_train), len(x_test), Counter(y_train), Counter(y_test))
over_s = SMOTE()
x_train, y_train = over_s.fit_resample(x_train, y_train)

y_train = np.array(y_train)

print("y_test",np.unique(y_test))
print(len(x_train), len(x_test), Counter(y_train), Counter(y_test))
model.fit(x_train, y_train)
start = time.time()
y_pred = model.predict(x_test)
end = time.time()
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("precision:", metrics.precision_score(y_test, y_pred,average="weighted"))
print("recall:", metrics.recall_score(y_test, y_pred,average="weighted"))
print("f1score:", metrics.f1_score(y_test, y_pred,average="weighted"))
print(metrics.confusion_matrix(y_test, y_pred))
cf=metrics.confusion_matrix(y_test, y_pred)
print(metrics.classification_report(y_test, y_pred))
print("predict time= ", end-start)
fmt = 'd'
sns.heatmap(cf, annot=True, cmap="Blues",fmt=fmt)
plt.xlabel("Predicted labels")
plt.ylabel("True labels")
plt.title("Confusion Matrix")
plt.show()


num_classes = cf.shape[0]
fpr_list = []
fnr_list = []
for i in range(num_classes):
    tp = cf[i, i]
    fn = np.sum(cf[i, :]) - tp
    fp = np.sum(cf[:, i]) - tp
    tn = np.sum(cf) - (tp + fn + fp)

    fpr = fp / (fp + tn) if (fp + tn) != 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) != 0 else 0

print(fpr)
print(fnr)
