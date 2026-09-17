import pandas as pd
from Data_preprocessing import data_process 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import pickle as pkl



# Anylsis Data_set with Pandas 
read_csv = pd.read_csv("data_set/Resume.csv")
print(read_csv.head())
print(read_csv.info())
print(read_csv.describe())
print(read_csv["Category"].value_counts())

# Filtering Data with Pandas 
find_missing_values = read_csv.isnull().sum()
print(f"Missing_values : {find_missing_values}")
find_duplicate_values = read_csv.duplicated().sum()
print(f"Duplicate Values : {find_duplicate_values}")

# Remove Duplicate_values
read_csv.drop_duplicates(inplace = True )

# remove  Useless Columns 
read_csv.drop(["ID","Resume_html"],axis = 1,inplace = True)
print(read_csv.columns)


# sample data for testing the fuction 
sample_string = r"I love it when professors draw a big question mark next to my answer on an exam because I’m always like yeah I don’t either ¯\_(ツ)_/¯ @VolphanCarol @littlewhitty @mysticalmanatee https://t.co/yZlafy0lsd <h1>hello</h1> 🤪 growthing referring teachings"

# filtered all data through data_process function 
read_csv["Filtered_data"] = read_csv["Resume_str"].apply(lambda x : data_process(x))
print(read_csv["Filtered_data"].head())

# encode the target column 
encoder = LabelEncoder()
read_csv["Category_label"]  = encoder.fit_transform(read_csv["Category"])
print(dict(zip(encoder.classes_, encoder.transform(encoder.classes_))))

# split data for training and testing 
X = read_csv[["Filtered_data"]]
y = read_csv["Category_label"]

x_train,x_test,y_train,y_test = train_test_split(X , y , random_state = 42 , test_size = 0.2 ,shuffle = True , stratify = y)

# apply TfidfVectorizer 
vector = TfidfVectorizer(ngram_range = (1,2),max_features=10000,sublinear_tf=True)
x_train_vector = vector.fit_transform(x_train["Filtered_data"])
x_test_vector = vector.transform(x_test["Filtered_data"])
print(x_test_vector.toarray())


# now apply model on data 
rfc_model = RandomForestClassifier(random_state=42)
rfc_model.fit(x_train_vector,y_train)
y_pred = rfc_model.predict(x_test_vector) 

# check model Performance
print(f"Accuracy_Score : {accuracy_score(y_test , y_pred)}")
print(f"Classification_Report : {classification_report(y_test , y_pred)}")
print(f"Confusion_Matrix : {confusion_matrix(y_test , y_pred)}")

# Apply LinearSVC Model 
ls_model = LinearSVC()
ls_model.fit(x_train_vector,y_train)
ls_pred = ls_model.predict(x_test_vector) 

# check model Performance
print(f"Accuracy_Score : {accuracy_score( y_test,ls_pred)}")
print(f"Classification_Report : {classification_report(y_test,ls_pred)}")
print(f"Confusion_Matrix : {confusion_matrix(y_test,ls_pred)}")


# Save models For use in apps
pkl.dump(vector,open("Models/Vector.pkl","wb"))
pkl.dump(ls_model,open("Models/Model.pkl","wb"))