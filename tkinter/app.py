from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from tkinter import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn import metrics


def plot_confusion_metrix(y_train,model_train):
    cm = metrics.confusion_matrix(y_train, model_train)
    plt.figure(1)
    plt.clf()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
    classNames = ['Non-ASD','ASD']
    plt.title('Confusion Matrix')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    tick_marks = np.arange(len(classNames))
    plt.xticks(tick_marks, classNames)
    plt.yticks(tick_marks, classNames)
    s = [['TN','FP'], ['FN', 'TP']]
    for i in range(2):
        for j in range(2):
            plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]))
    plt.show()

def report_performance(model):

    model_test = model.predict(X_test)

    print("\n\nConfusion Matrix:")
    print("{0}".format(metrics.confusion_matrix(y_test, model_test)))
    print("")
    print("\n\nClassification Report: ")
    print(metrics.classification_report(y_test, model_test))
    
    # acc1.append([ m, metrics.classification_report(y_test, model_test)])
    
    plot_confusion_metrix(y_test, model_test)
    # cm = metrics.confusion_matrix(y_test, model_test)
    # show_confusion_matrix(cm, ["Non-ASD","ASD"])
    

total_fpr = {}
total_tpr = {}


def roc_curves(model):
    predictions_test = model.predict(X_test)
    fpr, tpr, thresholds = roc_curve(predictions_test,y_test)
    roc_auc = auc(fpr, tpr)
    total_fpr[str((str(model).split('(')[0]))] = fpr
    total_tpr[str((str(model).split('(')[0]))] = tpr
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()

data = pd.read_csv('csv_result-Autism_Data.csv')
data.contry_of_res = data.contry_of_res.astype('str')
data.contry_of_res = data.contry_of_res.str.lower()
data.contry_of_res = data.contry_of_res.str.replace("'", "")
data.contry_of_res = data.contry_of_res.str.strip()
data.relation = data.relation.replace(np.nan, 'unknown', regex=True)
data.relation = data.relation.astype('str')
data.relation = data.relation.str.lower()
data.relation = data.relation.str.replace("'", "")
data.relation = data.relation.str.strip()
data.ethnicity = data.ethnicity.replace(np.nan, 'unknown', regex=True)
data.ethnicity = data.ethnicity.astype('str')
data.ethnicity = data.ethnicity.str.lower()
data.ethnicity = data.ethnicity.str.replace("'", "")
data.ethnicity = data.ethnicity.str.strip()
data.age_desc = data.age_desc.replace(np.nan, 'unknown', regex=True)
data.age_desc = data.age_desc.astype('str')
data.age_desc = data.age_desc.str.lower()
data.age_desc = data.age_desc.str.replace("'", "")
data.age_desc = data.age_desc.str.strip()
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
            'A10_Score', 'age', 'gender', 'ethnicity', 'jundice', 'austim', 'contry_of_res', 'used_app_before', 'age_desc', 'relation']
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

top = Tk()

# You can set the geometry attribute to change the root windows size
top.geometry("800x700")  # You want the size of the app to be 500x500
top.resizable(0, 0)  # Don't allow resizing in the x or y direction
top.title('Zenith, Machine Learning Project')
top.option_add("*Button.Background", "black")
top.option_add("*Button.Foreground", "red")

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

Label(top, text="age").place(x=label_pos_x, y=label_pos_y)
label_pos_y += 20

Label(top, text="gender").place(x=label_pos_x, y=label_pos_y)
label_pos_y += 20

Label(top, text="ethnicity").place(x=label_pos_x, y=label_pos_y)
label_pos_y += 20

Label(top, text="jundice").place(x=label_pos_x, y=label_pos_y)
label_pos_y += 20

Label(top, text="austim").place(x=label_pos_x, y=label_pos_y)
label_pos_y += 20

Label(top, text="contry_of_res").place(x=label_pos_x, y=label_pos_y)
label_pos_y += 20

Label(top, text="used_app_before").place(x=label_pos_x, y=label_pos_y)
label_pos_y += 20

Label(top, text="age_desc").place(x=label_pos_x, y=label_pos_y)
label_pos_y += 20

Label(top, text="relation").place(x=label_pos_x, y=label_pos_y)
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

e14 = Entry(top)
e14.place(x=entry_pos_x, y=entry_pos_y)
entry_pos_y += 20

e15 = Entry(top)
e15.place(x=entry_pos_x, y=entry_pos_y)
entry_pos_y += 20

e16 = Entry(top)
e16.place(x=entry_pos_x, y=entry_pos_y)
entry_pos_y += 20

e17 = Entry(top)
e17.place(x=entry_pos_x, y=entry_pos_y)
entry_pos_y += 20

e18 = Entry(top)
e18.place(x=entry_pos_x, y=entry_pos_y)
entry_pos_y += 20

e19 = Entry(top)
e19.place(x=entry_pos_x, y=entry_pos_y)
entry_pos_y += 20

entryText = StringVar()
prediction_entry = Entry(top, textvariable=entryText, width=60)
prediction_entry.place(x=200, y=625)
entry_pos_y += 20


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
    A14_Score = float(e14.get())
    A15_Score = float(e15.get())
    A16_Score = float(e16.get())
    A17_Score = float(e17.get())
    A18_Score = float(e18.get())
    A19_Score = float(e19.get())

    prediction = knn.predict([[A1_Score, A2_Score, A3_Score, A4_Score, A5_Score, A6_Score, A7_Score, A8_Score, A9_Score,
                               A10_Score, A11_Score, A12_Score, A13_Score, A14_Score, A15_Score, A16_Score, A17_Score, A18_Score, A19_Score]])

    if prediction[0] == 0:
        entryText.set('Not affected by Autism')
    else:
        entryText.set('Yes, he is affected by autism')


def create_knn_graph():
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import KFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    features= ['A9_Score','A6_Score','A5_Score','A7_Score','A8_Score','A2_Score','A3_Score','A4_Score','A1_Score','A10_Score', 'ethnicity','age','relation','jundice','used_app_before','contry_of_res','gender']
    predicted= ['Class/ASD']

    X = data[features].values
    y = data[predicted].values
    # Create standardizer
    standardizer = StandardScaler()

    knnModel = KNeighborsClassifier ()

    # Create a pipeline that standardizes, then runs logistic regression
    pipeline = make_pipeline(standardizer, knnModel)

    best_score=0
    # Create k-Fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=0)


    # start_time = timer(None)
    # perform K-fold cross-validation
    scores = cross_val_score(pipeline, X_train, y_train, cv=kf, scoring='accuracy', n_jobs=-1) # n_jobs=-1# Use all CPU scores
    print ("K-Fold scores=",scores)
    # compute mean cross-validation accuracy
    score = np.mean(scores)
    print ("Mean=",score)

    # rebuild a model on the combined training and validation set
    KNN2 = KNeighborsClassifier().fit(X_train, y_train.ravel())
    # timer(start_time)
    m = 'K-Nearest Neighbor with features selection & K Fold (Lasso = 0.001)'
    roc_curves(KNN2)

    # report_performance(KNN2) 
    #accuracy(KNN2)


def create_knn_confusion_matrix():
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import KFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    features= ['A9_Score','A6_Score','A5_Score','A7_Score','A8_Score','A2_Score','A3_Score','A4_Score','A1_Score','A10_Score', 'ethnicity','age','relation','jundice','used_app_before','contry_of_res','gender']
    predicted= ['Class/ASD']

    X = data[features].values
    y = data[predicted].values
    # Create standardizer
    standardizer = StandardScaler()

    knnModel = KNeighborsClassifier ()

    # Create a pipeline that standardizes, then runs logistic regression
    pipeline = make_pipeline(standardizer, knnModel)

    best_score=0
    # Create k-Fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=0)


    # start_time = timer(None)
    # perform K-fold cross-validation
    scores = cross_val_score(pipeline, X_train, y_train, cv=kf, scoring='accuracy', n_jobs=-1) # n_jobs=-1# Use all CPU scores
    print ("K-Fold scores=",scores)
    # compute mean cross-validation accuracy
    score = np.mean(scores)
    print ("Mean=",score)

    # rebuild a model on the combined training and validation set
    KNN2 = KNeighborsClassifier().fit(X_train, y_train.ravel())
    # timer(start_time)
    m = 'K-Nearest Neighbor with features selection & K Fold (Lasso = 0.001)'
    # roc_curves(KNN2)

    report_performance(KNN2) 
    #accuracy(KNN2)


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
    A14_Score = float(e14.get())
    A15_Score = float(e15.get())
    A16_Score = float(e16.get())
    A17_Score = float(e17.get())
    A18_Score = float(e18.get())
    A19_Score = float(e19.get())
    from sklearn.ensemble import RandomForestClassifier

    RF_model = RandomForestClassifier()
    RF_model.fit(X_train, y_train.ravel())

    prediction = RF_model.predict([[A1_Score, A2_Score, A3_Score, A4_Score, A5_Score, A6_Score, A7_Score, A8_Score, A9_Score,
                               A10_Score, A11_Score, A12_Score, A13_Score, A14_Score, A15_Score, A16_Score, A17_Score, A18_Score, A19_Score]])

    if prediction[0] == 0:
        entryText.set('Not affected by Autism')
    else:
        entryText.set('Yes, he is affected by autism')


def create_rf_graph():
    #import KNeighborsClassifier
    import numpy as np
    from sklearn.neighbors import KNeighborsClassifier

    # Setup arrays to store training and test accuracies
    neighbors = np.arange(1, 30)
    train_accuracy = np.empty(len(neighbors))
    test_accuracy = np.empty(len(neighbors))
    for i, k in enumerate(neighbors):
        # Setup a knn classifier with k neighbors
        knn = KNeighborsClassifier(n_neighbors=k)
    # Fit the model
        knn.fit(X_train, y_train.ravel())
        knn.fit(X_test, y_test.ravel())
    # Compute accuracy on the training set
        train_accuracy[i] = knn.score(X_train, y_train.ravel())
        print("train_accuracy=", train_accuracy[i])
    # Compute accuracy on the test set
        test_accuracy[i] = knn.score(X_test, y_test)
        print("test_accuracy=", test_accuracy[i])

    # Generate plot

    plt.title('k-NN Varying number of neighbors')
    plt.plot(neighbors, test_accuracy, label='Testing Accuracy')
    plt.plot(neighbors, train_accuracy, label='Training accuracy')
    plt.legend()
    plt.xlabel('Number of neighbors')
    plt.ylabel('Accuracy')
    plt.show()


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
    A14_Score = float(e14.get())
    A15_Score = float(e15.get())
    A16_Score = float(e16.get())
    A17_Score = float(e17.get())
    A18_Score = float(e18.get())
    A19_Score = float(e19.get())
    LogRegModel1 = LogisticRegression().fit(X_train, y_train.ravel())

    prediction = LogRegModel1.predict([[A1_Score, A2_Score, A3_Score, A4_Score, A5_Score, A6_Score, A7_Score, A8_Score, A9_Score,
                               A10_Score, A11_Score, A12_Score, A13_Score, A14_Score, A15_Score, A16_Score, A17_Score, A18_Score, A19_Score]])

    if prediction[0] == 0:
        entryText.set('Not affected by Autism')
    else:
        entryText.set('Yes, he is affected by autism')


def create_lr_graph():
    #import KNeighborsClassifier
    import numpy as np
    from sklearn.neighbors import KNeighborsClassifier

    # Setup arrays to store training and test accuracies
    neighbors = np.arange(1, 30)
    train_accuracy = np.empty(len(neighbors))
    test_accuracy = np.empty(len(neighbors))
    for i, k in enumerate(neighbors):
        # Setup a knn classifier with k neighbors
        knn = KNeighborsClassifier(n_neighbors=k)
    # Fit the model
        knn.fit(X_train, y_train.ravel())
        knn.fit(X_test, y_test.ravel())
    # Compute accuracy on the training set
        train_accuracy[i] = knn.score(X_train, y_train.ravel())
        print("train_accuracy=", train_accuracy[i])
    # Compute accuracy on the test set
        test_accuracy[i] = knn.score(X_test, y_test)
        print("test_accuracy=", test_accuracy[i])

    # Generate plot

    plt.title('k-NN Varying number of neighbors')
    plt.plot(neighbors, test_accuracy, label='Testing Accuracy')
    plt.plot(neighbors, train_accuracy, label='Training accuracy')
    plt.legend()
    plt.xlabel('Number of neighbors')
    plt.ylabel('Accuracy')
    plt.show()


Button(top, text="Confusion matrix using KNN", command=create_knn_confusion_matrix, activebackground="pink",
       activeforeground="blue").place(x=50, y=470)

Button(top, text="Classify using KNN", command=classify_using_knn, activebackground="pink",
       activeforeground="blue").place(x=230, y=470)

Button(top, text="Create KNN Graphs", command=create_knn_graph, activebackground="pink",
       activeforeground="green").place(x=490, y=470)

Button(top, text="Classify using Random Forest", command=classify_using_rf, activebackground="pink",
       activeforeground="blue").place(x=230, y=520)

Button(top, text="Create Random Forest Graphs", command=create_rf_graph, activebackground="pink",
       activeforeground="green").place(x=490, y=520)

Button(top, text="Classify using Logistic Regression", command=classify_using_lr, activebackground="pink",
       activeforeground="blue").place(x=230, y=570)

Button(top, text="Create Logistic Regression Graphs", command=create_lr_graph, activebackground="pink",
       activeforeground="green").place(x=490, y=570)

top.mainloop()
