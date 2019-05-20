from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from tkinter import *
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn import metrics

data = pd.read_csv(r'csv_result-Autism_Data.csv')
n_records = len(data.index)
n_asd_yes = len(data[data['Class/ASD'] == 'YES'])
n_asd_no = len(data[data['Class/ASD'] == 'NO'])
yes_percent = float(n_asd_yes) / n_records * 100

data.replace("?", np.nan, inplace=True)
total_missing_data = data.isnull().sum().sort_values(ascending=False)
percent_of_missing_data = (
    data.isnull().sum()/data.isnull().count()*100).sort_values(ascending=False)
missing_data = pd.concat(
    [
        total_missing_data,
        percent_of_missing_data
    ],
    axis=1,
    keys=['Total', 'Percent']
)
data.dropna(inplace=True)

gender_n = {"m": 1, "f": 0}
jundice_n = {"yes": 1, "no": 0}
austim_n = {"yes": 1, "no": 0}
used_app_before_n = {"yes": 1, "no": 0}
result_n = {"YES": 1, "No": 0}

# Encode columns into numeric
for column in data.columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])


features = ['A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score', 'A6_Score', 'A7_Score', 'A8_Score', 'A9_Score',
            'A10_Score','ethnicity','contry_of_res','relation']
predicted = ['Class/ASD']

X = data[features].values
y = data[predicted].values
split_test_size = 0.20

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=split_test_size, random_state=0)
# Setup a knn classifier with k neighbors
knn = KNeighborsClassifier()
# Fit the model
knn.fit(X_train, y_train.ravel())

from PIL import Image, ImageTk
top = Tk()

# You can set the geometry attribute to change the root windows size
top.geometry("700x700")  # You want the size of the app to be 500x500
top.resizable(0, 0)  # Don't allow resizing in the x or y direction
top.title('Autism Spectrum Disorder Classification Tool')
top.option_add("*Button.Background", "grey")
top.option_add("*Button.Foreground", "black")
from PIL import ImageTk, Image
photo = ImageTk.PhotoImage(Image.open(r'ribbon.png'))
logo = Label(top, image=photo)
logo.pack()
top.mainloop

label_pos_x = 30
label_pos_y = 50

Label(top, text="A1_Score").place(x=label_pos_x, y=label_pos_y)
label_pos_y += 20

Label(top, text="A2_Score").place(x=label_pos_x, y=label_pos_y)
label_pos_y += 20

Label(top, text="A3_Score").place(x=label_pos_x, y=label_pos_y)
label_pos_y += 20

Label(top, text="A4_Score").place(x=label_pos_x, y=label_pos_y)
label_pos_y += 20

Label(top, text="A5_Score").place(x=label_pos_x, y=label_pos_y)
label_pos_y += 20

Label(top, text="A6_Score").place(x=label_pos_x, y=label_pos_y)
label_pos_y += 20

Label(top, text="A7_Score").place(x=label_pos_x, y=label_pos_y)
label_pos_y += 20

Label(top, text="A8_Score").place(x=label_pos_x, y=label_pos_y)
label_pos_y += 20

Label(top, text="A9_Score").place(x=label_pos_x, y=label_pos_y)
label_pos_y += 20

Label(top, text="A10_Score").place(x=label_pos_x, y=label_pos_y)
label_pos_y += 20

Label(top, text="Ethnicity").place(x=label_pos_x, y=label_pos_y)
label_pos_y += 20

Label(top, text="Contry_of_res").place(x=label_pos_x, y=label_pos_y)
label_pos_y += 20

Label(top, text="Relation").place(x=label_pos_x, y=label_pos_y)
label_pos_y += 20

entry_pos_x = 200
entry_pos_y = 50

e1 = Entry(top)
e1.place(x=entry_pos_x, y=entry_pos_y)
entry_pos_y += 20

e2 = Entry(top)
e2.place(x=entry_pos_x, y=entry_pos_y)
entry_pos_y += 20

e3 = Entry(top)
e3.place(x=entry_pos_x, y=entry_pos_y)
entry_pos_y += 20

e4 = Entry(top)
e4.place(x=entry_pos_x, y=entry_pos_y)
entry_pos_y += 20

e5 = Entry(top)
e5.place(x=entry_pos_x, y=entry_pos_y)
entry_pos_y += 20

e6 = Entry(top)
e6.place(x=entry_pos_x, y=entry_pos_y)
entry_pos_y += 20

e7 = Entry(top)
e7.place(x=entry_pos_x, y=entry_pos_y)
entry_pos_y += 20

e8 = Entry(top)
e8.place(x=entry_pos_x, y=entry_pos_y)
entry_pos_y += 20

e9 = Entry(top)
e9.place(x=entry_pos_x, y=entry_pos_y)
entry_pos_y += 20

e10 = Entry(top)
e10.place(x=entry_pos_x, y=entry_pos_y)
entry_pos_y += 20

e11 = Entry(top)
e11.place(x=entry_pos_x, y=entry_pos_y)
entry_pos_y += 20

e12 = Entry(top)
e12.place(x=entry_pos_x, y=entry_pos_y)
entry_pos_y += 20

e13 = Entry(top)
e13.place(x=entry_pos_x, y=entry_pos_y)
entry_pos_y += 20

entryText = StringVar()
prediction_entry = Entry(top, textvariable=entryText, width=60)
prediction_entry.place(x=200, y=460)
entry_pos_y += 20

entryAccuracy = StringVar()
accuracy_entry = Entry(top, textvariable=entryAccuracy, width=60)
accuracy_entry.place(x=200, y=460+20)
entry_pos_y += 20

Label(top, text="List of features:").place(x=label_pos_x, y=label_pos_y)
label_pos_y += 20

label_pos_x = 400
label_pos_y = 20

Label(top, text="Classifier Performance:").place(x=label_pos_x, y=label_pos_y)
label_pos_y += 20

label_pos_x = 100
label_pos_y = 460

Label(top, text="Classification:").place(x=label_pos_x, y=label_pos_y)
label_pos_y += 20

Label(top, text="Accuracy:").place(x=label_pos_x, y=label_pos_y)
label_pos_y += 20

acc= []
acc1 = []
total_accuracy = {}
def accuracy(model):
    pred = model.predict(X_test)
    pred = (pred > 0.5)
    accu = metrics.accuracy_score(y_test,pred)
    errors = abs(pred - y_test)
    print('Model Performance')
    print("\nAccuracy Of the Model: ",accu)
    entryAccuracy.set('Accuracy Of the Model: '+ str(accu))
    print("\nAverage Error: {:0.2f} degrees.".format(np.mean(errors)))
    total_accuracy[str((str(model).split('(')[0]))] = accu
    
    model_test = model.predict(X_test)
    
     # true negative, false positive, etc...
    cm = confusion_matrix(y_test, model_test)
    
    total1=sum(sum(cm))
    
#confusion matrix calculate sensitivity,specificity

    specificity1 = cm[0,0]/(cm[0,0]+cm[0,1])
    print('Specificity Of the Model: ', specificity1,'\n' )

    sensitivity1 = cm[1,1]/(cm[1,0]+cm[1,1])
    print('Sensitivity Of the Model: ', sensitivity1,'\n')
    
    acc.append([accu,sensitivity1, specificity1])

def classify_using_knn():
    A1_Score = float(e1.get())
    A2_Score = float(e2.get())
    A3_Score = float(e3.get())
    A4_Score = float(e4.get())
    A5_Score = float(e5.get())
    A6_Score = float(e6.get())
    A7_Score = float(e7.get())
    A8_Score = float(e8.get())
    A9_Score = float(e9.get())
    A10_Score = float(e10.get())
    A11_Score = float(e11.get())
    A12_Score = float(e12.get())
    A13_Score = float(e13.get())

    best_grid = KNeighborsClassifier(n_neighbors=14, metric = 'hamming')
    best_grid.fit(X_train, y_train.ravel())
    
    prediction = knn.predict([[A1_Score, A2_Score, A3_Score, A4_Score, A5_Score, A6_Score, A7_Score, A8_Score, A9_Score,
                               A10_Score, A11_Score, A12_Score, A13_Score]])
    accuracy(best_grid)
    #entryText1.set()
    
    if prediction[0] == 0:
        entryText.set('Not pre-diagnose with ASD')
    else:
        entryText.set('Pre-diagnosed with ASD, seek clinician for further assistance')

def classify_using_rf():
    A1_Score = float(e1.get())
    A2_Score = float(e2.get())
    A3_Score = float(e3.get())
    A4_Score = float(e4.get())
    A5_Score = float(e5.get())
    A6_Score = float(e6.get())
    A7_Score = float(e7.get())
    A8_Score = float(e8.get())
    A9_Score = float(e9.get())
    A10_Score = float(e10.get())
    A11_Score = float(e11.get())
    A12_Score = float(e12.get())
    A13_Score = float(e13.get())
    
    from sklearn.ensemble import RandomForestClassifier

    RF3 = RandomForestClassifier(n_estimators=64 ,min_samples_split=12,min_samples_leaf=2,max_features=13,max_depth= 11, bootstrap=True)
    RF3.fit(X_train, y_train.ravel())

    prediction = RF3.predict([[A1_Score, A2_Score, A3_Score, A4_Score, A5_Score, A6_Score, A7_Score, A8_Score, A9_Score,
                               A10_Score, A11_Score, A12_Score, A13_Score]])
    accuracy(RF3)
    
    if prediction[0] == 0:
        entryText.set('Not pre-diagnose with ASD')
    else:
        entryText.set('Pre-diagnosed with ASD, seek clinician for further assistance')

def classify_using_lr():
    from sklearn.linear_model import LogisticRegression
    A1_Score = float(e1.get())
    A2_Score = float(e2.get())
    A3_Score = float(e3.get())
    A4_Score = float(e4.get())
    A5_Score = float(e5.get())
    A6_Score = float(e6.get())
    A7_Score = float(e7.get())
    A8_Score = float(e8.get())
    A9_Score = float(e9.get())
    A10_Score = float(e10.get())
    A11_Score = float(e11.get())
    A12_Score = float(e12.get())
    A13_Score = float(e13.get())

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import KFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

        # Create standardizer
    standardizer = StandardScaler()

        # Create logistic regression
    for c in [0.00001, 0.0001, 0.001, 0.1, 1, 10]:
        logRegModel = LogisticRegression(C=c)

        # Create a pipeline that standardizes, then runs logistic regression
        pipeline = make_pipeline(standardizer, logRegModel)

        best_score=0
        # Create k-Fold cross-validation
        kf = KFold(n_splits=5, shuffle=True, random_state=42)


        # perform K-fold cross-validation
        scores = cross_val_score(pipeline, X_train, y_train, cv=kf, scoring='accuracy', n_jobs=-1)
            
        # compute mean cross-validation accuracy
        score = np.mean(scores)
            
        # Find the best parameters and score
        if score > best_score:
            best_score = score
            best_parameters = c
            
    LogRegModel = LogisticRegression().fit(X_train, y_train.ravel())

    prediction = LogRegModel.predict([[A1_Score, A2_Score, A3_Score, A4_Score, A5_Score, A6_Score, A7_Score, A8_Score, A9_Score,
                               A10_Score, A11_Score, A12_Score, A13_Score]])

    accuracy(LogRegModel)
    
    if prediction[0] == 0:
        entryText.set('Not pre-diagnose with ASD')
    else:
        entryText.set('Pre-diagnosed with ASD, seek clinician for further assistance')

Button(top, text="Classify using K-Nearest Neigbhbors", command=classify_using_knn, activebackground="pink",
       activeforeground="blue").place(x=200, y=420)
Button(top, text="Classify using Random Forest", command=classify_using_rf, activebackground="pink",
       activeforeground="blue").place(x=200, y=385)
Button(top, text="Classify using Logistic Regression", command=classify_using_lr, activebackground="pink",
       activeforeground="blue").place(x=200, y=350)

top.mainloop()
