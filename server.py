from torchvision.models import alexnet, AlexNet_Weights
from flask import Flask, request, jsonify
from PIL import Image
from torchvision import models, transforms
import torch.nn as nn
import torch.optim as optim
import torch
import io

# Caricamento delle informazioni salvate
loaded_info = torch.load("modello_fia10.pth", map_location=torch.device("cpu"))

# Caricamento alexnet
alexnet = alexnet(weights=None)

# Modifica del primo strato convoluzionale
alexnet.features[0] = nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2)

# Modifica dell'ultimo strato fully-connected per la classificazione binaria
alexnet.classifier[6] = nn.Linear(4096, 1)

# Aggiunta della funzione di attivazione sigmoide all'ultimo strato
alexnet.classifier.add_module("7", nn.Sigmoid())

# Congelamento dei pesi dei primi strati
for param in alexnet.features.parameters():
    param.requires_grad = False

# Caricamento stato modello
alexnet.load_state_dict(loaded_info["model_state_dict"])

# Ricreazione dello stato dell'ottimizzatore
optimizer = optim.Adam(alexnet.parameters(), lr=0.001)
optimizer.load_state_dict(loaded_info["optimizer_state_dict"])

# Ricreazione del criterio di perdita
criterion = nn.BCELoss()  # o qualsiasi altro criterio usato
criterion.load_state_dict(loaded_info["criterion_state_dict"])

# Recupero delle altre informazioni
epoch = loaded_info["epoch"]
class_labels = loaded_info["class_labels"]
preprocessing_params = loaded_info["preprocessing"]

# Metti il modello in modalità di valutazione
alexnet.eval()

# Configurazione flask
app = Flask(__name__)


# Funzione per impostare gli header CORS
@app.after_request
def after_request(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
    response.headers.add("Access-Control-Allow-Methods", "POST")
    return response

# Route per la predizione
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "Nessun file inviato"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Nessun file selezionato"}), 400

    if file:
        image = Image.open(io.BytesIO(file.read()))
        input_tensor = process_input(image)

        with torch.no_grad():
            output = alexnet(input_tensor)

        predicted_probability = output.item()

        response = "Non Demented" if predicted_probability > 0.5 else "Demented"

        print(f'Result: {response} | Predicted_probability: {predicted_probability}')

        return jsonify({"prediction": response})


def process_input(image):
    # Applica le trasformazioni necessarie
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229]),
        ]
    )
    return transform(image).unsqueeze(0)  # Aggiungi una dimensione batch


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", ssl_context=("cert.pem", "key.pem"))
