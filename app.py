from pycaret.regression import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np

model = load_model('deployment_loan_prediction')

def predict(model, input_df):
    predictions_df = predict_model(estimator=model, data=input_df)
    predictions = predictions_df['Label'][0]
    return predictions

def run():

    from PIL import Image
    image = Image.open('logo.png')
    image_hospital = Image.open('loan.png')

    st.image(image,use_column_width=False)

    add_selectbox = st.sidebar.selectbox(
    "How would you like to predict?",
    ("Online", "Batch"))

    st.sidebar.info('This app is created to predict if the applicant should be granted a loan or not.')
    st.sidebar.success('https://www.pycaret.org')
    
    st.sidebar.image(image_hospital)

    st.title("Loan approval Prediction App")

    if add_selectbox == 'Online':
        
        Loan_ID= '0'
        Gender = st.selectbox('Gender', ['Female', 'Male'])
        Dependents = st.selectbox('Dependents', ['0', '3+', '2', '1'])
        Education = st.selectbox('Education', ['Graduate', 'Not Graduate'])
        ApplicantIncome= st.number_input('ApplicantIncome', min_value=150, max_value=81000, value=3856)
        CoapplicantIncome = st.number_input('CoapplicantIncome', min_value=0, max_value=41667, value=1229)
        LoanAmount= st.number_input('LoanAmount', min_value=17, max_value=700, value=126)
        Loan_Amount_Term = st.selectbox('Loan_Amount_Term', [360., 300., 180.,  84., 480., 240., 120.,  36.,  12.,  60.])
        Credit_History = st.selectbox('Credit_History', [0., 1.])
        Property_Area = st.selectbox('Property_Area', ['Semiurban', 'Rural', 'Urban'])
       
        if st.checkbox('Married'):
            Married = 'Yes'
        else:
            Married = 'No'
           
        if st.checkbox('Self_Employed'):
            Self_Employed = 'Yes'
        else:
            Self_Employed = 'No'
       

        output=""

    input_dict = {'Loan_ID':Loan_ID,'Gender' : Gender, 'Married' : Married, 'Dependents' : Dependents, 'Education' : Education, 'Self_Employed' : Self_Employed, 'ApplicantIncome' : ApplicantIncome, 'CoapplicantIncome' : CoapplicantIncome, 'LoanAmount': LoanAmount, 'Loan_Amount_Term' :Loan_Amount_Term, 'Credit_History': Credit_History, 'Property_Area':Property_Area}
    input_df = pd.DataFrame([input_dict])
    if st.button("Predict"):
        output = predict(model=model, input_df=input_df)
        output = str(output)
        if output=='1':
            output= 'Loan should be granted'
        else:
            output= 'Do not grant Loan'

    st.success('The output is {}'.format(output))

    if add_selectbox == 'Batch':

        file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])

        if file_upload is not None:
            data = pd.read_csv(file_upload)
            predictions = predict_model(estimator=model,data=data)
            st.write(predictions)

if __name__ == '__main__':
    run()