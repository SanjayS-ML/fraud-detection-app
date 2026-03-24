import streamlit as st
import pickle

model=pickle.load(open('model.pkl','rb'))
vectorizer = pickle.load(open('vectorizer.pkl','rb'))

st.set_page_config(page_title='Fraud Detector', layout='centered')
st.title('Fraud Detection App')
st.markdown('### 🔍 Detect whether a message is Fraud or Not')
st.write('Enter a message to check whether it is Fraud or Not')

#example button
if st.button('Try Example'):
    user_input = st.write("Congratualtions! You've won a free prize. Click now!")
else:
    user_input = st.text_area('Enter Text')

if st.button('Predict'):
    if user_input.strip() == "":
        st.warning('⚠️Please enter some text')
    else:
        #Transform input 
        input_data = vectorizer.transform([user_input])

        #predict probability
        prob = model.predict_proba(input_data)[0][1]

        threshold = 0.3
        prediction = 1 if prob > threshold else 0 

        if prediction ==1:
            st.error(f'⚠️Fraud Detected ')
            st.write(f'Confidence Score: {prob:.2f}')
        else:
            st.success(f'✅Not Fraud ')
            st.write(f'Confidence: {1-prob:.2f}')

# Divider 
st.markdown('----')

#Info Section 
st.markdown('### ℹ️ About this App')
st.write(''' This application uses Machine Learning to detect fraudulent messages.

- Model: Random Forest Classifier  
- Text Processing: TF-IDF Vectorization  
- Threshold tuning applied for better fraud detection  

Built using Streamlit for real-time predictions.
''')

#warning 
st.warning('⚠️ This is a demo ML model and may not be 100% accurate.')

