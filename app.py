from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('forex_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        close1 = float(request.form['close1'])
        close2 = float(request.form['close2'])
        close3 = float(request.form['close3'])
        close4 = float(request.form['close4'])
        close5 = float(request.form['close5'])

        input_features = np.array([[close1, close2, close3, close4, close5]])
        prediction = model.predict(input_features)

        signal = 'BUY ✅' if prediction[0] == 1 else 'SELL ❌'

        return render_template('result.html', prediction_text=f'Model Decision: {signal}')

    except Exception as e:
        return f"Error: {e}"

if __name__ == '__main__':
    app.run(debug=True)
