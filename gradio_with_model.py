import gradio as gr

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

from sklearn.linear_model import LogisticRegression

import joblib
import pandas as pd
import os

from input_schema import ChurnInput
from pydantic import ValidationError

# Required environment variables to run the application on KATE
#prefix = os.environ["KOD_PREFIX"]
port = 7860

# Load our Logistic Regression bundle (model+metadata) with PCA
bundle = joblib.load("model/logreg_pca.pkl")

# Extract the model (here actually a pipeline)
model = bundle['model']

# Generate input components from the Pydantic schema (ChurnInput)
field_names = list(ChurnInput.model_fields.keys())

# Create the inputs and tell Gradio that they're numbers
inputs = [
    gr.Number(label=field.replace("_", " ").title())
    for field in field_names
]

# Define the prediction function (we're type hinting that it will return a string)
# We don't need to do this but it helps to remind us what the output will be
def predict(*args) -> str:
    # Match predicts input with the field names 
    input_data = dict(zip(field_names, args))
    
    # Use Pydantic to validate the data
    try:
        validated = ChurnInput(**input_data)
    except ValidationError as e:
        # This makes Gradio display the error
        raise gr.Error(str(e)) 
    
    # Use Pydantic's model_dump to get a dict → DataFrame
    row = pd.DataFrame([validated.model_dump()])
    
    # Get a prediction
    pred = model.predict(row)[0]
    
    # Return the corresponding label
    return "Churn" if pred == 1 else "No churn"


# Launch the interface
demo = gr.Interface(
    fn=predict,
    inputs=inputs,
    outputs="label",
    title="Churn Prediction",
    description="Enter customer details to predict whether they are likely to churn."
)


# The section below will run the application
if __name__ == "__main__":
    demo.launch(
        server_name='0.0.0.0',
        server_port=port,
        #root_path=f'{prefix}/vscode/proxy/{port}',
    )

# ----------------
# EXTRAS
# Add in as needed
# ----------------


# Example: log predictions to a CSV
# ---------------------------------------------------------------------------------
# def predict(*args) -> str:
#     input_data = dict(zip(field_names, args))
#     validated = ChurnInput(**input_data)
#     row = pd.DataFrame([validated.model_dump()])
#     pred = model.predict(row)[0]
#     row["prediction"] = pred
#     log_path = "logs/predictions.csv"
    
#     # If the file exists don't write a header
#     write_header = not os.path.exists(log_path)
#     row.to_csv("logs/predictions.csv", mode="a", index=False, header=write_header)
#     return "Churn" if pred == 1 else "No churn"
# ---------------------------------------------------------------------------------


# Example: return probability instead of label
# ---------------------------------------------------------------------------------
# def predict(*args) -> str:
#     input_data = dict(zip(field_names, args))
#     validated = ChurnInput(**input_data)
#     row = pd.DataFrame([validated.model_dump()])
#     prob = model.predict_proba(row)[0][1]
#     return f"Churn probability: {prob:.2f}"
# ---------------------------------------------------------------------------------



# Example: return label and probability, using a configurable threshold
# ---------------------------------------------------------------------------------
# def predict(*args) -> tuple[str, str, float]:
#     input_data = dict(zip(field_names, args[:-1])) # Extra arg for the threshold
#     validated = ChurnInput(**input_data)
#     row = pd.DataFrame([validated.model_dump()])
#     prob = model.predict_proba(row)[0][1]
#     threshold = float(args[-1])
#     label = "Churn" if prob >= threshold else "No churn"
#     return label, f"{prob:.2f}", threshold
# inputs +=  [gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.5, label="Threshold")]
# outputs = [
#     gr.Label(label="Prediction"),
#     gr.Textbox(label="Predicted Probability"),
#     gr.Number(label="Threshold")]
# Also need to add `outputs=outputs,` to Interface
# ---------------------------------------------------------------------------------


# Example: use examples
# --------------------------------------------------------------------------------- examples = [
#     [113.0, 0, 0, 0, 317.1, 97.0, 53.91, 224.8, 70.0, 19.11, 215.1, 90.0, 9.68, 13.6, 2.0, 3.67, 1],
#     [92.0, 0, 0, 0.0, 139.8, 98.0, 23.77, 174.9, 143.0, 14.87, 201.6, 135.0, 9.07, 9.4, 7.0, 2.54, 2]
# ]
# Also need to add `examples=examples,` to Interface
