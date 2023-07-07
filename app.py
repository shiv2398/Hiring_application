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
    

def check_grade_format(grade):
    """
    Checks if the grade format is '86/100' or '86.0/100' and converts it to a string.
    If the format is incorrect, it raises an error.

    Parameters:
    - grade (str): The input grade string.

    Returns:
    - str: The converted grade string in the format '86/100' or '86.0/100' or an empty string if the format is incorrect.
    """
    if grade != '':
        if not re.match(r'^\d+(\.\d+)?/\d+$', grade):
            st.error("Grade should be in the format of '86/100' or '86.0/100'.")
            return ''
        else:
            return str(grade)
    else:
        return None




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

    # Input fields
    python = st.number_input('On a scale of 0-3, how much do you know Python:', min_value=0, max_value=3)
    ml = st.number_input('On a scale of 0-3, how much do you know ML:', min_value=0, max_value=3)
    dl = st.number_input('On a scale of 0-3, how much do you know DL:', min_value=0, max_value=3)
    nlp = st.number_input('On a scale of 0-3, how much do you know NLP:', min_value=0, max_value=3)
    avail = st.selectbox('Are you available for 3 months:', ['yes', 'no'])

    grade_10 = st.text_input('10th grade (e.g., 86.0/100):')
    grade_12 = st.text_input('12th grade (e.g., 86.0/100):')
    und_grade_post = st.text_input('graduation grade (e.g., 86.0/100):')
    grade_post = st.text_input('Post-graduation grade (e.g., 86.0/100):')
    other_skills = st.text_area('Do you have other skills (e.g., python, datascience):')

    # Convert input values to appropriate types
    python = int(python) if str(python).isdigit() else None
    ml = int(ml) if str(ml).isdigit() else None
    dl = int(dl) if str(dl).isdigit() else None
    nlp = int(nlp) if str(nlp).isdigit() else None
    avail = map_sentence_to_binary(avail)
    cr = 85

    # Validate and process grade inputs
    grade_10 = check_grade_format(grade_10)
    grade_12 = check_grade_format(grade_12)
    und_grade_post = check_grade_format(und_grade_post)
    grade_post = check_grade_format(grade_post)
    grad_list = [grade_10, grade_12, und_grade_post, grade_post]
    other_skills = str(other_skills) if isinstance(other_skills, str) else None

    degree = st.text_input('What is your degree (Bachelors/Masters (with stream)):')
    flag=False
    if st.button('Predict'):

        # Check if all inputs are provided
        if any(x is None for x in [python, ml, dl, nlp, avail, grade_10, grade_12, und_grade_post, grade_post, other_skills]):
            st.error("Please enter values for all the inputs before predicting.")
        elif any(x == '' for x in grad_list):
            st.error("Please enter valid values for the grade inputs.")
        elif other_skills is None or other_skills.strip() == '':
            st.error("Please enter values for the other skills.")
        elif degree.strip() == '':
            st.error("Please enter a value for the degree.")
        else:
            flag=True

        score = None
        degree_level = 0

        # Determine degree level based on keywords
        if any(keyword.lower().replace(' ', '') in degree.lower() for keyword in graduation_keywords):
            degree_level = 1
        elif any(keyword.lower().replace(' ', '') in degree.lower() for keyword in masters_keywords):
            degree_level = 2

        # Process other skills
        skills_list = [skill.strip().lower() for skill in other_skills.split(',')]
        invalid_skills = [skill for skill in skills_list if skill not in criteria]

        if len(invalid_skills) > 0:
            score = 0
            # st.error("Invalid skills entered. Skills should be in the format of 'python, machine learning, other skills'.")
        else:
            score = 1

        # Map grade values to a common scale
        grade_10 = grade_mapping_value(grade_10, cr)
        grade_12 = grade_mapping_value(grade_12, cr)
        und_grade_post = grade_mapping_value(und_grade_post, cr)
        grade_post = grade_mapping_value(grade_post, cr)
        if flag:
            # Prepare features array
            features = [python, ml, dl, nlp, avail, grade_10, und_grade_post, grade_12, grade_post, score, degree_level]
            features = np.array(features).reshape(1, -1)  # Reshape to (1, 11)
            b_features = shaper([python, ml, nlp, dl, avail])

            st.write(features)
            
            # Call the predict_cluster_classification function
            best_prediction, predicted_cluster, predicted_classification = predict_cluster_classification(features, b_features)
            
            # Display the predictions
        
            st.write('Predicted with all features: Cluster Model:', predicted_cluster)
            st.write('Predicted probabilistic labels: Classification:', predicted_classification)
            st.write('Prediction with Best Features: Cluster Model:', best_prediction)

if __name__ == '__main__':
    main()
