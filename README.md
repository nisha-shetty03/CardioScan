# CardioScan — Cardiac Risk Assessment

A machine-learning powered web application that predicts the likelihood of heart disease based on 11 clinical parameters. Built with Flask and a Random Forest classifier trained on the [Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction).

---

## Demo
> ⚠️ **Disclaimer:** CardioScan is an educational tool only. It does not replace professional medical diagnosis or treatment. Always consult a licensed healthcare professional.

---

## Features

- **11 clinical input parameters** — age, sex, chest pain type, resting BP, cholesterol, fasting blood sugar, resting ECG, max heart rate, exercise-induced angina, ST depression (Oldpeak), and ST slope
- **Instant prediction** with model confidence score
- **Personalised health guidance** based on your specific readings
- **Find nearby cardiologists** via GPS + Google Maps integration
- Clean, accessible dark UI with real-time slider feedback

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python, Flask |
| ML Model | scikit-learn (Random Forest) |
| Frontend | HTML, Vanilla CSS, Vanilla JS |
| Model training | Jupyter Notebook (`aimodel.ipynb`) |

---

## Project Structure

```
nish03/
├── app.py                  # Flask app & prediction route
├── heart_model.pkl         # Trained Random Forest model
├── aimodel.ipynb           # Model training notebook
├── requirements.txt        # Python dependencies
├── templates/
│   └── index2.html         # Main UI (single-page)
└── static/
    └── new.css             # Additional styles
```

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
python app.py
```

Open your browser at **http://127.0.0.1:5000**

---

## Model Details

| Property | Value |
|----------|-------|
| Algorithm | Random Forest Classifier |
| Dataset | Heart Failure Prediction (Kaggle) |
| Training samples | 918 |
| Approximate accuracy | ~87% |
| Features | 11 clinical parameters |

The model encodes categorical features (sex, chest pain type, ECG result, exercise angina, ST slope) before prediction. See [`aimodel.ipynb`](aimodel.ipynb) for full training details.

---

## Input Parameters

| Parameter | Description |
|-----------|-------------|
| Age | Patient age in years |
| Sex | Biological sex (M/F) |
| ChestPainType | TA / ATA / NAP / ASY |
| RestingBP | Resting blood pressure (mmHg) |
| Cholesterol | Serum cholesterol (mg/dL) |
| FastingBS | Fasting blood sugar > 120 mg/dL (0/1) |
| RestingECG | Normal / ST / LVH |
| MaxHR | Maximum heart rate achieved |
| ExerciseAngina | Exercise-induced angina (Y/N) |
| Oldpeak | ST depression induced by exercise |
| ST_Slope | Slope of peak exercise ST segment (Up/Flat/Down) |

---

## License

This project is for educational purposes. 
Dataset credit: [fedesoriano on Kaggle](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction).
