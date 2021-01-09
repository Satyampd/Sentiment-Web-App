import streamlit as st
import pandas as  pd
import numpy as np 
import plotly.express as px
import plotly.graph_objects as go
import pickle
from TextPreprocess import text_cleaning
# import tensorflow as tf
# from transformers import BertTokenizer
# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# TODO: Add BERT Model!


def app():

    st.title("Sentiment Analysis Using Natural Language Processing")
    st.markdown("Note: We have used IMDB Movie dataset  to train our model!")

    option = st.selectbox(
    'Please choose Machine Learning model for your predication',
    ('Logistic Regression', 'Naive Bayes'))   #, 'State of the Art - BERT'

    tfidf=pickle.load(open('tfidf_vector.pkl', 'rb')) 

    lr=pickle.load(open('logit_model.pkl', 'rb'))
    nb=pickle.load(open('nb_model.pkl', 'rb'))
    # BERT=pickle.load(open('bert_model.pkl', 'rb'))
    
    model=lr   # by default it would be LR only
    if option=='Logistic Regression':
        st.subheader('This is a linear model for classification, it tries to find a plane between point to classify, read more about Logistic Regression in Wikipedia') 
        model=lr
    elif option=='Naive Bayes':
        st.subheader('This is a simple probablistic based model based on Bayes Theorem, read more about Naive Bayes in Wikipedia')
        model=nb
    # else:
    #     st.subheader('BERT: Bidirectional Encoder Representations from Transformers is SOTA technique in filed of NLP, here we have used BERT w/ the help of HuggingFace library, read more about BERT in Wikipedia')
    #     model=BERT

    user_input = st.text_area("Please write the text here for knowing the Sentiment!! :)", "")

    if st.button('Predict Sentiment!'):
        with st.spinner("Working on your Text!"):
            text=text_cleaning(user_input)
            t=tfidf.transform([text]) 
            op=model.predict(t)
            op_prob=model.predict_proba(t)
            # else:
            #     text=text_cleaning(user_input)   
            #     bert_text = tokenizer(text, max_length=128, padding=True, truncation=True, return_tensors='tf')   # we are tokenizing before sending into our trained model
            #     output = model(bert_text)  
            #     tf_prediction=tf.nn.softmax(output[0], axis=-1) 
            #     op_prob=tf_prediction
            #     op=tf.argmax(tf_predictions, axis=1)
            #     op=label.numpy()[0]

            if op[0]==0:
                st.subheader('This is a Negative Text and Score is '+str( round(max((op_prob[0]))*100, 2)))
            else:    
                st.subheader('This is a Positive Text and Score is '+str( round(max((op_prob[0]))*100, 2)))

        fig=go.Figure(data=[go.Bar(x = ['Negative', 'Positive'],y=op_prob[0], marker={'color':['red','green']})])
        fig.update_layout(autosize=False ,plot_bgcolor='rgb(275, 275, 275)')
        fig.data[0].marker.line.width = 3
        fig.data[0].marker.line.color = "black" 
        st.plotly_chart(fig)


    st.subheader("""This whole project is made using 6 open-source libraries sklearn, NLTK, pandas, numpy, plotly and streamlit.
            Via - Satyampd(Username for Github, Kaggle and LinkedIn)""")    


if __name__ == "__main__":
	app()