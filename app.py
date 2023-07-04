import streamlit as st
import pickle 
import json
import pandas as pd
import numpy as np
import re
import warnings
warnings.filterwarnings('ignore')
criteria = ['R Programming', 'MongoDB', 'Flask', 'Data Analytics', 'MySQL', 'NoSQL', 'CI/CD', 'C++ Programming',
            'Data structure', 'Flutter', 'Robotic Process Automation (RPA)', 'OpenCV',
            'Database Management System (DBMS)', 'DBMS', 'Image Processing', 'Hadoop', 'Artificial Intelligence',
            'AI', 'Tableau', 'DSA']
# making an criteria for the graduation keywords and post_graduate keywords to filter the columns value 
# for graduation value will be 1 and for the other post graduation value will be 2
graduation_keywords=[
 'B.E Computer Science and Engineering (Artificial Intelligence and machine Learning)',
 'B.Tech',
 'B.Tech (Hons.)',
 'BS in Data Science and Applications',
 'Bachelor Of Science (B.Sc)',
 'Bachelor Of Technology (B.Tech) CS',
 'Bachelor of Computer Applications (BCA)',
 'Bachelor of Computer Applications (BCA) (Hons.)',
 'Bachelor of Computer Engineering',
 'Bachelor of Computer Science (B.C.S.)',
 'Bachelor of Information Technology (B.I.T.)',
 'Bachelor of Science (B.Sc)',
 'Bachelor of Science (B.Sc) (Hons.)',
 'Bachelor of Science (B.Sc) (Pass)',
 'Bachelors of Data Science',
 'Bsc',
 'Btech',
 'Certified Data Scientist',
 'Data Science',
]
masters_keywords=[
 'Integrated B.E & M.Tech',
 'Integrated B.Tech',
 'Integrated M.Sc.',
 'Integrated M.Tech',
 'Integrated MCA',
 'M.Sc. in Data Science',
 'M.Tech',
 'MCA',
 'Master of Computer Applications (MCA)',
 'Master of Computer Science (M.C.S.)',
 'Master of Data Science',
 'Master of Information Technology (M.I.T.)',
 'Master of Science (M.Sc) (Tech)',
 'Master of Technology (M.Tech)',
 'Ms In Data Science',
 'PG Diploma in Data Science',
 'Post Graduate Diploma In Computer Applications (P.G.D.C.A.)',
 'Post Graduate Diploma In Data Analytics And Machine Learning',
 'Post Graduate Diploma in Big Data Analytics (PG-DBDA)'
 ]
def grade_mapping_value(value, criteria):
    try:
        modified_value = 1 if (not pd.isna(value) and float(value.split('/')[0]) > criteria) else 0
    except (ValueError, AttributeError):
        modified_value = 0  # Set to 0 if value is not in expected format or is missing

    return modified_value
# Define a custom mapping function
def map_sentence_to_binary(value):
    if 'yes' in value.lower():
        return 1
    elif 'no' in value.lower():
        return 0
    else:
        return None  # or any other default value you prefer
def grade_checker(g):
    if isinstance(g,int):
        return 0
    x = g.strip()
    if not re.match(r'^\d+(\.\d+)?/\d+$', x):
        return 0
    else:
        return g


def predict_cluster_classification(features,best_feature):
    with open('kmeans_model.pkl', 'rb') as f:
        k_meansmodel = pickle.load(f)
    with open('cls_model.pkl', 'rb') as f:
        cls_model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('cls_scaler.pkl', 'rb') as f:
        cls_scaler = pickle.load(f)
    with open('cluster_names.json', 'r') as f:
        cluster_names = json.load(f)
    with open('bestfeatures_model.pkl', 'rb') as f:
        best_feature_model = pickle.load(f)


    cls_features_scaler = cls_scaler.transform(features)
    cluster_features = scaler.transform(features)

    predicted_clustering = k_meansmodel.predict(cluster_features)
    predicted_classification = cls_model.predict(cls_features_scaler)
    predict_best=best_feature_model.predict(best_feature)

    predicted_cluster_name = cluster_names[str(predicted_clustering[0])]
    predicted_classification_name = cluster_names[str(predicted_classification[0])]
    predicted_best_prediction_name = cluster_names[str(predict_best[0])]

    return predicted_best_prediction_name,predicted_cluster_name, predicted_classification_name
def shaper(x):
    return np.array([int(i) for i in x]).reshape(1, -1)
def main():
    st.set_page_config(page_title='Intern Selection Process', layout='wide')
    st.title('Best Intern Selection App')

    python = st.number_input('On a scale of 0-3, how much do you know Python:', min_value=0, max_value=3)
    ml = st.number_input('On a scale of 0-3, how much do you know ML:', min_value=0, max_value=3)
    dl = st.number_input('On a scale of 0-3, how much do you know DL:', min_value=0, max_value=3)
    nlp = st.number_input('On a scale of 0-3, how much do you know NLP:', min_value=0, max_value=3)
    avail = st.selectbox('Are you available for 3 months:', ['yes', 'no'])

    grade_10 = st.text_input('10th grade (e.g., 86.0/100):')
    grade_12 = st.text_input('12th grade(e.g., 86.0/100):')
    und_grade_post = st.text_input('graduation grade(e.g., 86.0/100):')
    grade_post = st.text_input('Post-graduation grade(e.g., 86.0/100):')
    other_skills = st.text_area('Do you have other skills(e.g python,datascience):')

    python = int(python) if str(python).isdigit() else 0
    ml = int(ml) if str(ml).isdigit() else 0
    dl = int(dl) if str(dl).isdigit() else 0
    nlp = int(nlp) if str(nlp).isdigit() else 0

    avail = map_sentence_to_binary(avail)
    avail = str(avail) if isinstance(avail, str) else '0'
    cr = 80

    grade_10 = grade_checker(grade_mapping_value(grade_10, cr))
    grade_12 = grade_checker(grade_mapping_value(grade_12, cr))
    grade_post = grade_checker(grade_mapping_value(grade_post, cr))
    und_grade_post = grade_checker(grade_mapping_value(und_grade_post,cr))
    other_skills = str(other_skills) if isinstance(other_skills, str) else ''

    score = 0
    skill = other_skills.lower()
    # Check if the skill matches any criteria in the list
    for criteria_skill in criteria:
        if criteria_skill.lower() in skill:
            score = 1
            break

    degree = st.text_input('What is your degree(Bachelors/Masters(with stream)):')
    
    degree_level = 0
    if isinstance(degree, str):
        if any(keyword.lower().replace(' ', '') in degree for keyword in graduation_keywords):
            degree_level = 1
        elif any(keyword.lower().replace(' ', '') in degree for keyword in masters_keywords):
            degree_level = 2

    features = [python, ml, dl, nlp, avail, grade_10, und_grade_post, grade_12, grade_post, score, degree_level]
   # ['python', 'ML', 'NLP', 'DL', 'availability']
    b_features=[python,ml,nlp,dl,avail]
   
    features = np.array([int(i) for i in features]).reshape(1, -1)

    b_features=shaper(b_features)

    if st.button('Predict'):
        best_prediction,predicted_cluster, predicted_classification = predict_cluster_classification(features,b_features)
        st.write('Predicted with all features     :  Cluster Model  :  ', predicted_cluster)
        st.write('Predicted probabilistic labels  :  Classification :  ', predicted_classification)
        st.write('Prediction with Best Features   :  Cluster Model  :  ',best_prediction)

if __name__ == '__main__':
    main()

