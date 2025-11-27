## Fake Job Posting Detector

This project helps users detect whether a job posting is real or potentially fake using machine learning, job index verification, email safety checks, and scam keyword detection.

Render Deployment Link:
https://fake-job-detector-sdfr.onrender.com/ 


### 1. Overview

Fake job scams are increasing, and many people lose money or personal information.
This tool analyzes job postings and gives:

	•	A Real % Score
	•	A Fake % Score
	•	A clear final verdict
	•	Verification insights (Adzuna API, email checks, keywords).

Users simply enter:

	•	Job Title
	•	Job Description
	•	Optional Company
	•	Optional Location


### 2. Features

#### Machine Learning Model

Uses a trained Logistic Regression model with TF-IDF to classify job postings.

Public Job Index Verification (Adzuna API)

Checks if the job exists on trusted job boards.
If found, it increases the chance of the job being real.

##### Email and Domain Safety Check

Extracts emails from the job description and checks for:
	•	Free email domains (gmail.com, yahoo.com, etc.)
	•	Disposable/temporary domains
	•	Company domain mismatch

#### Scam Keyword Detection

Flags dangerous terms such as:

	•	“quick money”
	•	“wire transfer”
	•	“training fee”
	•	“pay to start”
	•	“no interview”
	•	“gift card”

#### Final Combined Confidence

All checks (model + verification signals) combine into:

	•	Final Real %
	•	Final Fake %
	•	Explanations

### Modern UI (Streamlit)

Smooth, responsive, and visually appealing with gradient headers and insight cards.

## 3. Project Structure
fake-job-detector/
│
├── app/
│   └── app.py                 # Streamlit UI
│
├── utils/
│   ├── verifier.py            # Adzuna API, email checks, keyword detection
│   └── text_cleaning.py       # Text preprocessing functions
│
├── models/
│   ├── fake_job_model.pkl     # Trained ML model
│   └── tfidf_vectorizer.pkl   # Vectorizer for text features
│
├── notebooks/                 # Model training & analysis notebooks
│
├── data/                      # Dataset used for training/testing
│
├── requirements.txt           # Python dependencies
│
└── README.md                  # Project documentation

## 4. How It Works

#### Step 1: Cleaning

Job text is cleaned and normalized.

#### Step 2: Model Prediction

TF-IDF + ML model returns a fake probability.

#### Step 3: Verification Checks
	•	Adzuna search
	•	Email safety
	•	Scam keywords

#### Step 4: Combine Signals

Calculates final real vs fake confidence.

#### Step 5: Display Results

UI shows risk meter, percentages, insights, and explanations.

### 5. Installation

#### Install dependencies
pip install -r requirements.txt

#### Run the app 
streamlit run app/app.py

## 6. Inputs Needed

Users can enter:

	•	Job Title
	•	Job Description
	•	Company Name (optional)
	•	Job Location (optional)

More details give more accurate results.


## 7. Outputs Provided

The system displays:

	•	Final Real Score
	•	Final Fake Score
	•	Adzuna match info
	•	Email risk analysis
	•	Scam keyword results.

## 8. Purpose

This project is built to:

	•	Help job seekers avoid scams
	•	Detect unsafe or suspicious job posts
	•	Demonstrate AI + verification techniques

Created as part of the Fall 2025 Master’s Project at UNH.

⸻

### 9. Author

Raju Kotturi.

Master of Information Technology

University of New Hampshire
