import streamlit as st
import numpy as np
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import pandas as pd
import pickle
from keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Churn Predictor", page_icon="🔮", layout="centered")
## load the trained model
model=load_model('models/model.keras')

with open('models/onehot_encoder_geo.pkl','rb') as file:
    label_encoder_geo=pickle.load(file)

with open('models/label_encoder_gender.pkl','rb') as file:
    label_encoder_gender=pickle.load(file)

with open('models/scaler.pkl','rb') as file:
    scaler=pickle.load(file)

with open("models/feature_columns.pkl", "rb") as f:
    feature_order = pickle.load(f)

def load_training_data():
    return pd.read_csv('data/training_data.csv')

training_data=load_training_data()

st.markdown("""
      <style>
            .stApp{
                background-color: #0e1117;
                color: #ffffff;
            }
            .title-style{
                text-align: center;
                font-size:3em;
                font-weight: bold;
                color: #ff4b4b;
                margin-top: 10px;
                margin-bottom: 30px;  
            }
            .input-container{
               background-color: #1e1e1e;
               padding: 30 px;
            border-radius: 15 px;
            box-shadow: 0 4px 12 px rgba(0,0,0,0.4);
            }
            .stButton{
                display: flex;
                justify-content: center;
            }
            div.stButton > button:first-child{
                background-color: #ff4b4b;
                color: white;
                border: none;
                padding: 0.6em 2em;
                font-size: 16px;
                transition: background-color 0.3s ease;
            }
            div.stButton > button:first-child:hover{
                background-color: #ff1e1e;
            }
            """,unsafe_allow_html=True)

## streamlit app
st.markdown("<h1 class='title-style'>🔮Customer Churn Prediction</h1>",unsafe_allow_html=True)

st.markdown("<div class='input-container'>",unsafe_allow_html=True)
geography=st.selectbox('Geography',label_encoder_geo.categories_[0])
gender=st.selectbox('Gender',label_encoder_gender.classes_)
age=st.slider('Age',18,92)
balance=st.number_input('Balance')
credit_score=st.number_input('Credit Score')
estimated_salary=st.number_input('Estimated Salary')
tenure=st.slider('Tenure',0,10)
num_of_products=st.slider('Number of Products')
has_cr_card=st.selectbox('Has Credit Card',[0,1])
is_active_member=st.selectbox('Is Active Member',[0,1])

st.markdown("</div>",unsafe_allow_html=True)
##Prepare the input data
if st.button("🚀 Predict Churn"):
    input_data=pd.DataFrame({
        'CreditScore':[credit_score],
        'Gender':label_encoder_gender.transform([gender])[0],
        'Age':[age],
        'Tenure':[tenure],
        'Balance':[balance],
        'NumOfProducts':[num_of_products],
        'HasCrCard':[has_cr_card],
        'IsActiveMember':[is_active_member],
        'EstimatedSalary':[estimated_salary]
    })

    geo_encoded=label_encoder_geo.transform(pd.DataFrame({'Geography':[geography]}))
    geo_encoded_df=pd.DataFrame(geo_encoded,columns=label_encoder_geo.get_feature_names_out(['Geography']))


    input_data=pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)
    input_data = input_data[feature_order]
    input_data_scaled=scaler.transform(input_data)

    prediction=model.predict(input_data_scaled)
    prediction_proba=prediction[0][0]

    st.subheader(f'📊Churn Probability: {prediction_proba:.2f}')
    if(prediction_proba>0.5):
        st.error('⚠️ The customer is **likely to churn**')
    else:
        st.success('✅The customer is not likely to churn')
    

    st.markdown("---")
    st.subheader("📈 Feature Distributions with Your Input")

    
    plt.figure(figsize=(6, 4))
    sns.histplot(training_data['Age'], kde=True, bins=30, color='skyblue', edgecolor='black')
    plt.axvline(age, color='red', linestyle='--', label='Given Input')
    plt.title(f'Distribution of Age')
    plt.xlabel('Age')
    plt.ylabel('Density')
    plt.legend()
    st.pyplot(plt.gcf())
    plt.clf()

    counter=training_data['CreditScore'].value_counts()
    counter=counter.head(10)
    labels=counter.index
    values=counter.values
    plt.figure(figsize=(10,8))
    plt.pie(values,labels=labels,autopct='%1.1f%%')
    plt.title('Pie chart for top 10 categories of Credit Score')
    plt.text(0, -1.2, f'Given Input: {int(credit_score)}', ha='center', color='red', fontsize=10)
    st.pyplot(plt.gcf())
    plt.clf()

    plt.figure(figsize=(6, 5))
    sns.boxplot(y=training_data['Balance'], color='skyblue')
    plt.scatter(0, balance, color='red', label=f'Given Input: {balance:.0f}', zorder=5)
    plt.title('Box Plot of Balance')
    plt.ylabel('Balance')
    plt.legend()
    st.pyplot(plt.gcf())
    plt.clf()

