from flask import Flask, request, render_template
from PyPDF2 import PdfReader
import re
import pickle

app = Flask(__name__)

# Load models
rf_classifier_categorization = pickle.load(open('models/rf_classifier_categorization.pkl', 'rb'))
tfidf_vectorizer_categorization = pickle.load(open('models/tfidf_vectorizer_categorization.pkl', 'rb'))
rf_classifier_job_recommendation = pickle.load(open('models/rf_classifier_job_recommendation.pkl', 'rb'))
tfidf_vectorizer_job_recommendation = pickle.load(open('models/tfidf_vectorizer_job_recommendation.pkl', 'rb'))

# Clean resume text
def cleanResume(txt):
    txt = re.sub('http\S+\s', ' ', txt)
    txt = re.sub('RT|cc', ' ', txt)
    txt = re.sub('#\S+\s', ' ', txt)
    txt = re.sub('@\S+', '  ', txt)
    txt = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', txt)
    txt = re.sub(r'[^\x00-\x7f]', ' ', txt)
    txt = re.sub('\s+', ' ', txt)
    return txt.strip()

def pdf_to_text(file):
    reader = PdfReader(file)
    text = ''
    for page in reader.pages:
        text += page.extract_text() or ''
    return text

def predict_category(resume_text):
    resume_text = cleanResume(resume_text)
    resume_tfidf = tfidf_vectorizer_categorization.transform([resume_text])
    category = rf_classifier_categorization.predict(resume_tfidf)[0]
    return category

def job_recommendation(resume_text):
    resume_text = cleanResume(resume_text)
    resume_tfidf = tfidf_vectorizer_job_recommendation.transform([resume_text])
    job = rf_classifier_job_recommendation.predict(resume_tfidf)[0]
    return job

# Extractors
def extract_contact_number(text):
    pattern = r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
    match = re.search(pattern, text)
    return match.group() if match else 'Not found'

def extract_email(text):
    pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    match = re.search(pattern, text)
    return match.group() if match else 'Not found'

def extract_skills(text):
    skills_list = [
        'Python', 'Data Analysis', 'Machine Learning', 'SQL', 'Java', 'C++', 'JavaScript',
        'HTML', 'CSS', 'React', 'Node.js', 'Git', 'Deep Learning', 'TensorFlow', 'Keras',
        'PyTorch', 'NLP', 'Computer Vision', 'Docker', 'AWS', 'Azure', 'GCP'
    ]
    found = []
    for skill in skills_list:
        if re.search(r'\b' + re.escape(skill) + r'\b', text, re.I):
            found.append(skill)
    return found if found else ['No skills found']

def extract_education(text):
    education_keywords = ['Bachelor', 'Master', 'B.Sc', 'M.Sc', 'PhD', 'Diploma', 'High School', 'Associate Degree']
    found = []
    for edu in education_keywords:
        if re.search(edu, text, re.I):
            found.append(edu)
    return found if found else ['No education info found']

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'resume' not in request.files:
            return render_template('index.html', error="No file part")

        file = request.files['resume']
        if file.filename == '':
            return render_template('index.html', error="No selected file")

        if file:
            try:
                resume_text = pdf_to_text(file)
                contact = extract_contact_number(resume_text)
                email = extract_email(resume_text)
                skills = extract_skills(resume_text)
                education = extract_education(resume_text)
                category = predict_category(resume_text)
                recommended_job = job_recommendation(resume_text)

                return render_template('index.html',
                    contact=contact,
                    email=email,
                    skills=skills,
                    education=education,
                    category=category,
                    recommended_job=recommended_job
                )
            except Exception as e:
                return render_template('index.html', error=f"Error processing file: {str(e)}")

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
