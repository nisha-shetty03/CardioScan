from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

model = joblib.load("heart_model.pkl")

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index2.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("\n" + "="*50)
        print("Form data received:", dict(request.form))
        
        age = int(request.form['age'])
        sex = request.form['sex']  # 'M' or 'F'
        chestpaintype = request.form['chestpaintype']  # 'ATA', 'NAP', 'ASY', 'TA'
        restingbp = int(request.form['restingbp'])
        cholesterol = int(request.form['cholesterol'])
        fastingbs = int(request.form['fastingbs'])  # 0 or 1
        restingecg = request.form['restingecg']  # 'Normal', 'ST', 'LVH'
        maxhr = int(request.form['maxhr'])
        exerciseangina = request.form['exerciseangina']  # 'Y' or 'N'
        oldpeak = float(request.form['oldpeak'])
        st_slope = request.form['st_slope']  # 'Up', 'Flat', 'Down'

       
        sex_encoded = 1 if sex == 'M' else 0
        
        chest_pain_map = {'ATA': 0, 'NAP': 1, 'ASY': 2, 'TA': 3}
        chestpaintype_encoded = chest_pain_map.get(chestpaintype, 0)
        
        restingecg_map = {'Normal': 0, 'ST': 1, 'LVH': 2}
        restingecg_encoded = restingecg_map.get(restingecg, 0)
        
        exerciseangina_encoded = 1 if exerciseangina == 'Y' else 0
        
        st_slope_map = {'Up': 0, 'Flat': 1, 'Down': 2}
        st_slope_encoded = st_slope_map.get(st_slope, 0)

       
        features = [[age, sex_encoded, chestpaintype_encoded, restingbp, 
                     cholesterol, fastingbs, restingecg_encoded, maxhr,
                     exerciseangina_encoded, oldpeak, st_slope_encoded]]
        
        print("Features:", features)
        print("Number of features:", len(features[0]))
        
        prediction = model.predict(features)
        print("Prediction:", prediction[0])
        print("="*50 + "\n")
        
        result_text = "Heart Disease Detected" if prediction[0] == 1 else "No Heart Disease Detected"

        probability = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(features)[0]
            probability = float(proba[int(prediction[0])])

        return jsonify({
            "success": True,
            "prediction": int(prediction[0]),
            "result": result_text,
            "probability": probability
        })

    except KeyError as e:
        print(f"\n❌ Missing field: {e}")
        return jsonify({
            "success": False,
            "error": f"Missing required field: {str(e)}"
        })
    
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\n")
        return jsonify({
            "success": False,
            "error": str(e)
        })

if __name__ == "__main__":
    app.run(debug=True)
