# Import necessary libraries
import streamlit as st
# import nltk
import spacy
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.sentiment import SentimentIntensityAnalyzer

# Step 1: Data Collection
# Code to collect resume data and store it in a structured format
resume_data = [
    "John Doe\nSoftware Engineer\nContact: john.doe@example.com\nEducation: Bachelor's Degree in Computer Science",
    "Jane Smith\nData Analyst\nContact: janesmith@example.com\nEducation: Master's Degree in Statistics",
    # Add more resume data here
]

# Step 2: Resume Analyzing
# Preprocess the resumes
nltk.download('punkt')
nltk.download('stopwords')
stopwords = set(stopwords.words('english'))

def preprocess_resume(resume_text):
    tokens = nltk.word_tokenize(resume_text)
    tokens = [token.lower() for token in tokens if token.isalpha() and token.lower() not in stopwords]
    return tokens

preprocessed_resumes = [preprocess_resume(resume) for resume in resume_data]

# Extract contact details
def extract_contact_details(resume_text):
    lines = resume_text.split('\n')
    contact_details = {}
    for line in lines:
        if 'contact' in line.lower():
            contact_details['contact'] = line.split(':')[1].strip()
        elif 'education' in line.lower():
            contact_details['education'] = line.split(':')[1].strip()
    return contact_details

contact_details = [extract_contact_details(resume) for resume in resume_data]

# Step 3: Skill Matching and Ranking
job_skills = ["python", "machine learning", "data analysis", "communication"]

matched_skills = [set(job_skills).intersection(set(skills)) for skills in preprocessed_resumes]

ranked_candidates = sorted(zip(resume_data, matched_skills), key=lambda x: len(x[1]), reverse=True)

# Step 4: Experience and Education Analysis
nlp = spacy.load('en_core_web_sm')

def extract_job_titles(resume_text):
    doc = nlp(resume_text)
    job_titles = []
    for ent in doc.ents:
        if ent.label_ == 'JOB_TITLE':
            job_titles.append(ent.text)
    return job_titles

job_titles = [extract_job_titles(resume) for resume in resume_data]

# Step 5: Keyword Extraction and Sentiment Analysis
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(resume_data)

analyzer = SentimentIntensityAnalyzer()

sentiments = [analyzer.polarity_scores(resume) for resume in resume_data]

# Step 6: Integration and User Interface with Streamlit
def main():
    st.title("Resume Screening App")
    
    # Display contact details
    st.header("Contact Details")
    for contact in contact_details:
        st.write(contact)
    
    # Display ranked candidates
    st.header("Ranked Candidates")
    for i, (resume, skills) in enumerate(ranked_candidates):
        st.subheader(f"Candidate {i+1}")
        st.write(resume)
        st.write("Matched Skills:", skills)
    
    # Display job titles
    st.header("Job Titles")
    for titles in job_titles:
        st.write(titles)
    
    # Display sentiments
    st.header("Sentiments")
    for sentiment in sentiments:
        st.write(sentiment)
    
if __name__ == "__main__":
    main()
