# Import packages
import pandas as pd
import numpy as np
import streamlit as st

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Dataframe
s = pd.read_csv("social_media_usage.csv")

# Clean SM
def clean_sm(x): 
    x = np.where (x == 1,
                  1,
                  0)
    return(x)



# Create SS dataframe
ss = pd.DataFrame({
    "sm_li":clean_sm(s.web1h),
    "income":np.where(s["income"] > 9, np.nan, s["income"]),
    "education":np.where(s["educ2"] > 8, np.nan, s["educ2"]),
    "parent":np.where(s["par"] == 1, 1, 0),
    "married":np.where(s["marital"] == 1, 1, 0),
    "female":np.where(s["gender"] == 2, 1, 0),
    "age":np.where(s["age"] > 97, np.nan, s["age"])
})

# Remove missing values
ss = ss.dropna()

# Create target vector and feature set
y = ss["sm_li"]
X = ss[["income", "education", "parent", "married", "female", "age"]]

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    stratify=y,       # same number of target in training & test set
                                                    test_size=0.2,    # hold out 20% of data for testing
                                                    random_state=987) # set for reproducibility

# Instantiate a logistic regression model and set weight to balanced
lr = LogisticRegression(class_weight = "balanced")

# Fit the model with the training data
lr.fit(X_train, y_train)

# Make predictions using the model
y_pred = lr.predict(X_test)

## Streamlit
with st.sidebar:
    inc = st.number_input("Income (low=1 to high=9)",1,9)
    deg = st.number_input("College degree (no=0)", 0,1)
    par = st.number_input("Parent (0=no, 1=yes)", 0, 1)
    mar = st.number_input("Married (0=no, 1=yes)", 0,1)
    fem = st.number_input("Female (0=no, 1=yes)", 0,1)
    age = st.number_input("Age (low=1 to high=100)", 1, 100)


# Create lables from numberic inputs

# Income
if inc <= 3:
    inc_label = "low income"
elif inc > 3 and inc < 7: 
    inc_label = "middle income"
else: 
    inc_label = "high income"

# Education
if deg == 1:
    deg_label = "college graduate"
else: 
    deg_label = "non-college graduate"

# Parental Status
if par == 1:
    par_label = "parent"
else:
    par_label = "non-parent"

# Marital
if mar == 1:
    mar_label = "married"
else:
    mar_label = "non-married"

# Female
if fem == 1:
    fem_label = "female"
else:
    fem_label = "non-female"

# Age
if age <= 18:
    age_label = "adolescent"
elif age > 18 and age < 40:
    age_label = "young adult"
elif age > 40 and age < 65:
    age_label = "middle-age adult"
else:
    age_label = "senior adult"
    
persona = [inc,deg,par,mar,fem,age]

predicted_sm_li = lr.predict([persona])

probs = lr.predict_proba([persona])
probs_r = probs[0][1]

st.header('Hi! Welcome to my project.')
st.subheader('This app predicts the likelihood that a person has a LinkedIn profile based on your answers.')

st.write(f"This {age_label} {mar_label} {par_label}, whom is a {inc_label} {deg_label}, has a {probs_r} probability of having a LinkedIn account")
st.write(f"Predicted class):{predicted_sm_li[0]}")

