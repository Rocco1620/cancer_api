from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
from fastapi import HTTPException

# Cargar el modelo
model = joblib.load('cancer_model.pkl')

# Inicializar la app
app = FastAPI()

# Definir el esquema de entrada (espera una lista de 30 números)
class InputData(BaseModel):
    data: list

# Ruta para predicción
@app.post("/predict")
def predict(input_data: InputData):
    # Validar que la entrada tenga 30 valores
    if len(input_data.data) != 30:
        raise HTTPException(
            status_code=400,
            detail=f"Se esperaban 30 valores, pero se recibieron {len(input_data.data)}"
        )

    # Validar que todos los elementos sean numéricos (int o float)
    if not all(isinstance(x, (int, float)) for x in input_data.data):
        raise HTTPException(
            status_code=400,
            detail="Todos los elementos deben ser numéricos (int o float)"
        )

    # Convertir a array Numpy
    X = np.array([input_data.data])

    # Predecir con el modelo
    prediction = model.predict(X)
    resultado = 'Maligno' if prediction[0] == 0 else 'Benigno'

    return {
        'predicción': int(prediction[0]),
        'diagnóstico': resultado
    }
