import streamlit as st
import pandas as pd
import pickle

st.write("""
# Here is your Result
""")
st.write("""
## This result just predict your anwser is in your heart
""")

st.sidebar.header('Infomation Input')
st.sidebar.header('This is for Consideration of your school line')
st.sidebar.subheader('Please enter your data:')



def get_input():
    #widgets
    p_sex = st.sidebar.radio('Sex', ['Male','Female'])
    p_gpaEng = st.sidebar.slider('GPA_Eng',0.000, 4.000, 1.000 )
    p_gpaMath = st.sidebar.slider('GPA_Math', 0.000, 4.000, 1.000 )
    p_gpaSci = st.sidebar.slider('GPA_Sci', 0.000, 4.000, 1.000 )
    p_gpaSoc = st.sidebar.slider('GPA_Sco', 0.000, 4.000, 1.000 )
    p_q3 = st.sidebar.selectbox('Q3', [0,1])
    p_q4 = st.sidebar.selectbox('Q4', [0,1])
    p_q6 = st.sidebar.selectbox('Q6', [0,1])
    p_q26 = st.sidebar.selectbox('Q26', [0,1])
    p_q28 = st.sidebar.selectbox('Q28', [0,1])

    

    #if p_sex == 'Male': p_sex = 'M'
    #else: p_sex = 'F'
    

    #dictionary
    #data = {'GPA_Eng': p_gpaEng,
            #'GPA_Math': p_gpaMath,
            #'GPA_Sci': p_gpaSci,
            #'GPA_Sco': p_gpaSoc,
            #'Sex': p_sex,
            #'Student_Th': p_stuTH ,
            #'Q3': p_q3,
            #'Q4': p_q4,
            #'Q6': p_q6,
            #'Q26': p_q26,
            #'Q28': p_q28,
    #}

    data = {'StudentTH': 1,
        'GPA_Eng': p_gpaEng,
        'GPA_Math': p_gpaMath,
        'GPA_Sci': p_gpaSci,
        'GPA_Sco': p_gpaSoc,
        'Q3': p_q3,
        'Q4': p_q4,
        'Q6': p_q6,
        'Q26': p_q26,
        'Q28': p_q28,
        'Sex': p_sex
    }

    #create data frame
    data_df = pd.DataFrame(data, index=[0])
    return data_df

df = get_input()
st.write(df)

data_sample =  pd.read_excel('new_sample_tcas.xlsx')
df = pd.concat([df, data_sample],axis=0)
new_num_data = df[['StudentTH','GPA_Eng','GPA_Math','GPA_Sci','GPA_Sco','Q3','Q4','Q6','Q26','Q28']]
cat_data = pd.get_dummies(df[['Sex']])

#Combine all transformed features together
X_new = pd.concat([new_num_data,cat_data], axis=1)
#new_num_data = new_num_data[:1]
X_new = X_new[:1] # Select only the first row (the user input data)
#Drop un-used feature



# -- Reads the saved normalization model
load_nor = pickle.load(open('normalization.pkl', 'rb'))
#Apply the normalization model to new data
X_new = load_nor.transform(X_new)
st.write(X_new)

#new_num_data = load_nor.transform(new_num_data)
#st.write(new_num_data)

# -- Reads the saved classification model
load_knn = pickle.load(open('best_knn.pkl', 'rb'))
# Apply model for prediction
#prediction = load_knn.predict(new_num_data)
prediction = load_knn.predict(X_new)
st.write(prediction)


