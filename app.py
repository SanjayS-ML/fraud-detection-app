import streamlit as st
import pickle

model=pickle.load(open('model.pkl','rb'))
vectorizer = pickle.load(open('vectorizer.pkl','rb'))

st.set_page_config(page_title='Fraud Detector', layout='centered')
st.title('Fraud Detection App')
st.write('Enter a message to check whether it is Fraud or Not')

user_input = st.text_area('Enter Text')

if st.button('Predict'):
    if user_input.strip() == "":
        st.warning('Please enter some text')
    else:
        #Transform input 
        input_data = vectorizer.transform([user_input])

        #predict probability
        prob = model.predict_proba(input_data)[0][1]

        threshold = 0.3
        prediction = 1 if prob > threshold else 0 

        if prediction ==1:
            st.error(f'Fraud Detected (Confidence: {prob:.2f})')
        else:
            st.success(f'Not Fraud (Confidence: {1-prob:.2f})')

