import pandas as pd 
from contractions import fix
import re
from emoji import demojize
import string
import nltk
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import pickle as pkl




# Download Bundels 
nltk.download("punkt")
nltk.download("stopwords")


# Decalration of stopwords and SnowballStemmer

stop_word = set(stopwords.words("english"))
stem  = SnowballStemmer(language="english")


# Anylsis Data_set with Pandas 
read_csv = pd.read_csv("data_set/Resume.csv")
read_csv = read_csv.sample(n = 1200 , random_state = 42  )
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

def data_process(text):
    # convert String into lower
    text = text.lower()
    # fix contraction 
    text = fix(text)
    # remove links from string 
    text = re.sub(r"http\S+","",text)
    # remove html tags from string 
    text = re.sub(r"<.*?>","",text)
    # filtering Emojis
    text = demojize(text,language="en",delimiters=(" "," ")) 
    text = text.replace("_"," ")
    # remove special character
    text = text.translate(str.maketrans(" " ," " ,string.punctuation))
    text = re.sub(r"[^a-zA-Z]\s","",text)
    # convert string into tokens
    text = word_tokenize(text)
    # Remove stop_words
    text = [word for word in text if word not in stop_word]
    # convert into original form 
    text = [stem.stem(word) for word in text]
    # convert into string 
    text = " ".join(text)
    return text


# sample data for testing the fuction 
sample_string = r"I love it when professors draw a big question mark next to my answer on an exam because I’m always like yeah I don’t either ¯\_(ツ)_/¯ @VolphanCarol @littlewhitty @mysticalmanatee https://t.co/yZlafy0lsd <h1>hello</h1> 🤪 growthing referring teachings"

# filtered all data through data_process function 
read_csv["Filtered_data"] = read_csv["Resume_str"].apply(lambda x : data_process(x))
print(read_csv["Filtered_data"].head())

# encode the target column 
encoder = LabelEncoder()
read_csv["Category_label"]  = encoder.fit_transform(read_csv["Category"])
print(read_csv[read_csv["Category"] == "TEACHER"])

# split data for training and testing 
X = read_csv[["Filtered_data"]]
y = read_csv["Category_label"]

x_train,x_test,y_train,y_test = train_test_split(X , y , random_state = 42 , test_size = 0.2 ,shuffle = True , stratify = y)

# apply TfidfVectorizer 
vector = TfidfVectorizer()
x_train_vector = vector.fit_transform(x_train["Filtered_data"])
x_test_vector = vector.transform(x_test["Filtered_data"])
print(x_test_vector.toarray())


# now apply model on data 
rfc_model = RandomForestClassifier(random_state=42)
rfc_model.fit(x_train_vector,y_train)
y_pred = rfc_model.predict(x_test_vector) 

# check model Performance
print(f"Accuracy_Score : {accuracy_score(y_pred  , y_test)}")
print(f"Classification_Report : {classification_report(y_pred  , y_test)}")
print(f"Confusion_Matrix : {confusion_matrix(y_pred  , y_test)}")

# Save models For use in apps
pkl.dump(vector,open("Models/Vector.pkl","wb"))
pkl.dump(rfc_model,open("Models/Model.pkl","wb"))