from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load('model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        Speed = float(request.form['Speed'])
        Vertical_Acceleration = float(request.form['Vertical_Acceleration'])
        Lateral_Acceleration = float(request.form['Lateral_Acceleration'])
        Longitudinal_Acceleration = float(request.form['Longitudinal_Acceleration'])
        Roll = float(request.form['Roll'])
        Pitch = float(request.form['Pitch'])
        Yaw = float(request.form['Yaw'])

        # Creating a pandas DataFrame with the input data
        input_data = pd.DataFrame({
            'Speed': [Speed],
            'Vertical_Acceleration': [Vertical_Acceleration],
            'Lateral_Acceleration': [Lateral_Acceleration],
            'Longitudinal_Acceleration': [Longitudinal_Acceleration],
            'Roll': [Roll],
            'Pitch': [Pitch],
            'Yaw': [Yaw]
        })

        # Making prediction
        prediction = model.predict(input_data)

        # Interpret the prediction
        # prediction_text = 'Safe' if prediction[0] == 2 else 'Unsafe'

        return jsonify({'prediction': str(prediction[0])}), 200
        # return render_template('index.html', prediction_text='Driver is {}'.format(prediction_text))
        # # Extract features from the form data
        # speed = float(request.form['speed'])
        # x_acc = float(request.form['x_acc'])
        # y_acc = float(request.form['y_acc'])
        # z_acc = float(request.form['z_acc'])
        # roll = float(request.form['roll'])
        # pitch = float(request.form['pitch'])
        # yaw = float(request.form['yaw'])

        # # Make prediction using the loaded model
        # prediction = model.predict([[speed, x_acc, y_acc, z_acc, roll, pitch, yaw]])

        # # output = round(prediction[0], 2)
        # # output = prediction[0]
        # return render_template('index.html', prediction_text='Driver is  $ {}'.format(prediction))

        # # Return the prediction as JSON response
        # #return jsonify({'prediction': str(prediction[0])}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)

