from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from model import *
app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/submit-form', methods=['POST'])
def submit_form():
    # Receive data from the form
    area = int(request.form['area'])
    bhk = int(request.form['bhk'])
    bathroom = int(request.form['bathroom'])
    is_furnished = int(request.form['isFurnished'])
    parking = int(request.form['parking']) if 'parking' in request.form else None
    is_apartment = int(request.form['isApartment'])

    # Process the data (example: just echoing back the data in this case)
    response_data = {
    }

    response_data['price'] =predict_price(area, bhk, bathroom, is_furnished, parking, is_apartment);

    # Return a JSON response
    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True)
