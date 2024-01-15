from flask import Flask, jsonify, request, render_template
import numpy as np
import pandas as pd
import pickle
import nltk
from nltk.corpus.reader import reviews

app = Flask(__name__)
valid_userid = ['00sab00','1234','zippy','zburt5','joshua','dorothy w','rebecca','walker557','samantha','raeanne','kimmie','cassie','moore222']

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    user_input=[str(x) for x in request.form.values()]
    user_input=user_input[0]
    #print(user_input)
    if user_input in valid_userid and request.method == 'POST':
        user_input=[str(x) for x in request.form.values()]
        user_input=user_input[0]
        #print(user_input)
        pickled_tfidf_vectorizer = pd.read_pickle('Tfidf_vectorizer.pkl')
        pickled_model = pd.read_pickle('Logistic_Reg_final_model.pkl')
        pickled_user_final_rating = pd.read_pickle('user_final_rating.pkl')        
        pickled_mapping = pd.read_pickle('prod_id_name_mapping.pkl') 
        pickled_reviews_data = pd.read_pickle('reviews_data_all_cols.pkl')

        #print(pickled_user_final_rating)
        
        #recommendations = pd.DataFrame(pickled_user_final_rating.loc[user_input]).reset_index()
        #recommendations.columns=['id','user_pred_rating']
        #recommendations= recommendations.sort_values(by='user_pred_rating',ascending=False)[0:20]
        

        recommendations = pd.DataFrame(pickled_user_final_rating.loc[user_input]).reset_index()
        recommendations.rename(columns={recommendations.columns[1]: "user_pred_rating" }, inplace = True)
        recommendations = recommendations.sort_values(by='user_pred_rating', ascending=False)[0:20]
       
        recommendations.rename(columns={recommendations.columns[0]: "prod_id" }, inplace = True)
        pickled_mapping.rename(columns={pickled_mapping.columns[0]: "prod_id" }, inplace = True)  
        pickled_reviews_data.rename(columns={pickled_reviews_data.columns[0]: "prod_id" }, inplace = True)
    
        recommendations = pd.merge(recommendations,pickled_mapping, left_on="prod_id", right_on="prod_id", how = "left")
        
        improved_recommendations= pd.merge(recommendations,pickled_reviews_data[['prod_id','clean']], left_on='prod_id', right_on='prod_id', how = 'left')
        test_data_for_user = pickled_tfidf_vectorizer.transform(improved_recommendations['clean'].values.astype('U'))
        
        sentiment_prediction_for_user = pickled_model.predict(test_data_for_user)
        sentiment_prediction_for_user = pd.DataFrame(sentiment_prediction_for_user, columns=['Predicted_Sentiment'])

        improved_recommendations= pd.concat([improved_recommendations, sentiment_prediction_for_user], axis=1)
        
        a=improved_recommendations.groupby('prod_id')
        b=pd.DataFrame(a['Predicted_Sentiment'].count()).reset_index()
        b.columns = ['prod_id', 'Total_reviews']        
        c=pd.DataFrame(a['Predicted_Sentiment'].sum()).reset_index()
        c.columns = ['prod_id', 'Total_predicted_positive_reviews']
        
        improved_recommendations_final=pd.merge( b, c, left_on='prod_id', right_on='prod_id', how='left')
        
        improved_recommendations_final['Positive_sentiment_rate'] = improved_recommendations_final['Total_predicted_positive_reviews'].div(improved_recommendations_final['Total_reviews']).replace(np.inf, 0)
        
        improved_recommendations_final= improved_recommendations_final.sort_values(by=['Positive_sentiment_rate'], ascending=False )
        improved_recommendations_final=pd.merge(improved_recommendations_final, pickled_mapping, left_on='prod_id', right_on='prod_id', how='left')
        
        name_display= improved_recommendations_final.head(5)
        name_display= name_display['name']
        
        output = name_display.to_list()
        output.insert(0,"***")
        output="***\t \t***".join(output)
        #print(output)
        return render_template('index.html', prediction_text='Top 5 recommendations are- {}'.format(output))
    elif not user_input in  valid_userid:
        return render_template('index.html',text='No Recommendation found for the user')
    else :
        return render_template('index.html')
    
if __name__ == '__main__':
    app.run(debug=True)

    #app.run(host='127.0.0.1', port=5000)
