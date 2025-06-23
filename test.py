
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st

# load data
data = pd.read_csv("creditcard.csv")

def load_data():
    # Replace data with the path to your CSV file
    data = pd.read_csv('creditcard.csv')
    return data

dt=load_data()
# separate legitimate and fraudulent transactions
legit = data[data.Class == 0]
fraud = data[data.Class == 1]

# undersample legitimate transactions to balance the classes
legit_sample = legit.sample(n=len(fraud), random_state=2)
data = pd.concat([legit_sample, fraud], axis=0)

# split data into training and testing sets
X = data.drop(columns="Class", axis=1)
y = data["Class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

# train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# evaluate model performance
train_acc = accuracy_score(model.predict(X_train), y_train)
test_acc = accuracy_score(model.predict(X_test), y_test)


# Function to add a background image
def add_background_from_url(image_url):
    """
    Add a background image to the Streamlit app from a URL.
    """
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url("{image_url}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Call the function with your image URL
add_background_from_url("https://aicontentfy.com/hubfs/Blog/e2f82ed6-4180-4648-9560-949a48793661.jpg")  # Replace with your image URL


# Web app
st.title("Credit Card Fraud Detection Model")
input_df = st.text_input('Enter All Required Features Values')
input_df_splited = input_df.split(',')

submit = st.button("Submit")

if submit:
    features = np.asarray(input_df_splited,dtype=np.float64)
    prediction = model.predict(features.reshape(1,-1))

    if prediction[0] == 0:
        st.markdown("<h1 style='text-align: centre; color:green;'>Legitimate Transaction</h1>", unsafe_allow_html=True)
    else:
        st.markdown("<h1 style='text-align: centre; color: red;'>Fraudulent Transaction</h1>", unsafe_allow_html=True)
        


dt=load_data()
st.write('**Source Data:**')
st.dataframe(dt)


st.write('**Processed data:**')
st.dataframe(data)


