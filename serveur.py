from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Charger le modèle depuis le fichier .pkl
with open('mon_modele.pkl', 'rb') as f:
    model = pickle.load(f)

# Endpoint pour prédiction
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    # Supposons que les données à prédire sont dans data['features']
    prediction = model.predict([data['features']])
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
