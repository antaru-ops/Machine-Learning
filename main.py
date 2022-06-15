import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
import seaborn as sns

st.title("Heart Disease")

st.write("""
# Explore different classifier
Which one is the best?
""")

ds_name = st.sidebar.selectbox("Select Dataset", ("Heart Disease", ""))

classifier_name = st.sidebar.selectbox("Select Classifier", ("Support Vector Machine", "Decision Tree"))

def get_dataset(ds_name):
    ds_name == "Heart Disease"
    #ds = pd.read_csv(r'C:\Users\andre\Desktop\Jupyter Notebook\Data\heart.csv')
    url=r"heart.csv"
    ds=pd.read_csv(url,header = 0)
    newds = ds.dropna()
    x = newds.iloc[:, :13]
    y = newds.iloc[:, -1]
    return x,y

x, y = get_dataset(ds_name)
st.write("Shape of Dataset", x.shape)
st.write("Number of Classes", len(np.unique(y)))

# plotting

#fig = plt.figure(figsize = (9,7))
#heart = sns.load_dataset(r'C:\Users\andre\Desktop\Jupyter Notebook\Data\heart.csv')
#sns.histplot(data=newds['age'], x="Age")
#sns.histplot(
#    data=heart,
#    x="age",
#    hue="target",
#    multiple="stack"
#)
#plt.title("Age")
#st.pyplot(fig)


def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == 'Support Vector Machine':
        C = st.sidebar.slider('C', 0.01, 10.0)
        params['C'] = C
    else:
        min_samples_leaf = st.sidebar.slider('min_samples_leaf', 1, 10)
        params['min_samples_leaf'] = min_samples_leaf
    return params

params = add_parameter_ui(classifier_name)

def get_classifier(clf_name, params):
    clf = None
    if clf_name == 'Support Vector Machine':
        clf = SVC(C=params['C'])
    else:
        clf = clf = DecisionTreeClassifier(min_samples_leaf = params['min_samples_leaf'],criterion = 'entropy', random_state = 42)
    return clf

clf = get_classifier(classifier_name, params)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.fit_transform(x_test)

clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

acc = accuracy_score(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred)

st.write(f'Classifier = {classifier_name}')
st.write(f'Accuracy =', acc)
st.write(f'Confusion Matrix ', cm)

TP, FP, FN, TN = confusion_matrix(list(y_test), list(y_pred), labels=[0, 1]).ravel()

st.write(f'True Postive ', TP)
st.write(f'True Negative ', TN)
st.write(f'False Postive ', FP)
st.write(f'False Negative ', FN)

# Precision 
precision =  TP / (TP + FP)
st.write(f'Precision = ', precision)

# Negative Predicted Value
npv = TN / (TN + FN)
st.write(f'Negative Predicted Value = ', npv)

# Sensitivity
sensitivity = TP / (TP + FN)
st.write(f'Sensitivity = ', sensitivity)

# Specificity 
specificity = TN / (TN + FP)
st.write(f'Specificity = ', specificity)

# Accuracy
accuracy = (TP + TN) / (TP + TN + FP + FN)
st.write(f'Accuracy = ', accuracy)

# F-score
f_score_no = 2 * precision * sensitivity / (precision + sensitivity)
st.write(f'F-score for No Heart Disease = ', f_score_no)

f_score_yes = 2 * npv * specificity / (npv + specificity)
st.write(f'F-score for With Heart Disease = ', f_score_yes)

pca = PCA(13)
x_projected = pca.fit_transform(x)

x1 = x_projected[:, 7] #thalach = maximum heart rate achieve
x2 = x_projected[:, -1]

fig = plt.figure()
plt.scatter(x1, x2, c=y, alpha=0.8, cmap='viridis')

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar()

#plt.show()
st.pyplot(fig)
