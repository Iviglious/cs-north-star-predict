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
from model_utils import load_case_data, add_file_month, DEMO_MONTHS


# Required environment variables to run the application on KATE
port = 7860

# Load the trained pipeline. Older artifacts may wrap it in a mapping.
loaded_model = joblib.load("model/assigned_team_pipeline.joblib")
model = loaded_model["model"] if isinstance(loaded_model, dict) else loaded_model

# Generate input components from the Pydantic schema (ChurnInput)
field_names = list(ChurnInput.model_fields.keys())

# Load the case data from the CSV into a DataFrame, keeping only the columns
# we need (case_id + the model input fields).
csv_path = "data/Q4-Dec.csv"
cases_df_org = pd.read_csv(csv_path, dtype=str)
cases, repair_log = load_case_data("data")
cases = add_file_month(cases)

cases_df = cases[
    cases["file_month"].isin(DEMO_MONTHS)
].sort_values("file_month").drop_duplicates("case_id", keep="last").copy()

# Keep only the case_id, the fields the schema/model expects, and the
# actual assigned_team (the "expected" answer we compare the prediction to).
needed_columns = ["case_id"] + field_names + ["assigned_team"]
cases_df = cases_df[needed_columns]


print(cases_df.columns)
#print(cases_df_org[needed_columns].columns)
#cases_df = cases_df_org[needed_columns]

# Index by case_id so we can quickly look up a row by the selected case.
cases_by_id = cases_df.set_index("case_id")

# List of all case ids to populate the dropdown.
case_ids = cases_df["case_id"].tolist()

# Create inputs that match each field's expected value type.
categorical_options = {
    "channel": ["email", "phone", "in_app", "webchat"],
    "priority": ["Low", "Medium", "High", "Urgent"],
    "plan_tier": ["Free", "Standard", "Pro", "Enterprise"],
    "sentiment": ["Positive", "Neutral", "Negative"],
}

# Confidence percentage slider withd default value of 75%
confidence_slider = gr.Slider(minimum=0, maximum=100, value=75, label="Confidence Percentage")


def build_input(field_name):
    """Create the matching Gradio component for a single schema field."""
    label = field_name.replace("_", " ").title()
    field_info = ChurnInput.model_fields[field_name]
    default = None if field_info.is_required() else field_info.get_default()

    if field_name in categorical_options:
        return gr.Dropdown(
            choices=categorical_options[field_name],
            value=default,
            label=label,
            allow_custom_value=True,
        )
    elif field_name == "tags":
        return gr.Textbox(value=default, label=label)
    else:
        return gr.Number(value=default, label=label)


def coerce_field_value(field_name, raw_value):
    """Convert a raw CSV string value into the type each component expects."""
    #print(f"Coercing field '{field_name}' with raw value: {raw_value}")
    missing = pd.isna(raw_value)
    if not hasattr(missing, "__len__") and missing:
        return None
    # Numeric (gr.Number) fields: everything that isn't categorical or tags.
    if field_name not in categorical_options and field_name != "tags":
        try:
            return float(raw_value)
        except (TypeError, ValueError):
            return None
    if field_name == "tags" and isinstance(raw_value, list):
        return ";".join(str(tag) for tag in raw_value)
    return str(raw_value)


def load_case(case_id):
    """Return the field values for the selected case plus its expected team."""
    if case_id is None or case_id not in cases_by_id.index:
        # Nothing selected: leave the inputs unchanged and clear the expected team.
        return [gr.update() for _ in field_names] + [gr.update(value=None)]

    row = cases_by_id.loc[case_id]
    field_updates = [
        gr.update(value=coerce_field_value(field_name, row[field_name]))
        for field_name in field_names
    ]
    expected_team = None if pd.isna(row["assigned_team"]) else str(row["assigned_team"])
    return field_updates + [gr.update(value=expected_team)]


def comparison_html(predicted, expected, top_predictions, confidence_percentage):
    """Render a green (Correct) / red (Incorrect) comparison badge."""
    if not expected:
        return ""
    is_correct = str(predicted).strip().lower() == str(expected).strip().lower()
    color = "#22c55e" if is_correct else "#ef4444"
    text = "Correct" if is_correct else "Incorrect"
    return (
        f"<div style='background-color:{color};color:white;padding:10px;"
        f"border-radius:8px;text-align:center;font-weight:bold;font-size:16px;'>"
        f"{text}</div>"

        # Check if the top prediction is below the confidence threshold and display a warning if so
        + (
            f"<div style='background-color:#facc15;color:black;padding:10px;"
            f"border-radius:8px;text-align:center;font-weight:bold;font-size:16px;'>"
            f"Warning: The top prediction is below the confidence threshold."
            f"</div>"
            if top_predictions and list(top_predictions.values())[0] < (confidence_percentage / 100.0)
            else ""
        )

        # Table of top predictions and their probabilities
        + f"<table style='width:100%;margin-top:10px;border-collapse:collapse;'>"
        f"<tr><th style='text-align:left;padding:5px;border-bottom:1px solid #ddd;'>Top Predictions</th>"
        f"<th style='text-align:right;padding:5px;border-bottom:1px solid #ddd;'>Probability</th></tr>"
        + "".join(
            f"<tr><td style='padding:5px;border-bottom:1px solid #ddd;'>{team}</td>"
            f"<td style='padding:5px;border-bottom:1px solid #ddd;text-align:right;'>{prob:.2%}</td></tr>"
            for team, prob in top_predictions.items()
        )
        + "</table>"
    )


# Define the prediction function. It returns the predicted team label and a
# colored badge comparing it against the expected assigned team.
def predict(*args) -> tuple:
    # The last argument is the expected assigned team (not a model input).
    *field_values, expected_team, confidence_percentage = args

    # Match predicts input with the field names
    input_data = dict(zip(field_names, field_values))

    # Use Pydantic to validate the data
    try:
        validated = ChurnInput(**input_data)
    except ValidationError as e:
        # This makes Gradio display the error
        raise gr.Error(str(e))

    # Use Pydantic's model_dump to get a dict → DataFrame
    row_data = validated.model_dump()
    row_data["tags"] = [
        tag.strip() for tag in row_data["tags"].split(";") if tag.strip()
    ]
    row = pd.DataFrame([row_data])

    # Get a prediction
    pred = model.predict(row)[0]
    proba = model.predict_proba(row)[0]

    # Convert the model.classes_ and probabilities into a dictionary for easier display
    # Show the top 3 predicted classes with their probabilities
    top_indices = proba.argsort()[-3:][::-1]
    top_classes = [model.classes_[i] for i in top_indices]
    top_probabilities = [proba[i] for i in top_indices]

    # Create a dictionary of the top classes and their probabilities
    top_predictions = dict(zip(top_classes, top_probabilities))

    # Return the predicted label and the comparison badge
    # Pass the dictionary of top classes and probabilities to the comparison_html function for display
    return pred, comparison_html(pred, expected_team, top_predictions, confidence_percentage)


# Build the interface with Blocks so the case dropdown can populate the fields.
with gr.Blocks(title="Team CShaNTy - Northstar Desk - Assign Team Prediction") as demo:
    gr.Markdown("# Team CShaNTy - Northstar Desk - Assign Team Prediction")
    gr.Markdown(
        "Select a case to auto-fill its details, or edit the fields manually, "
        "then predict the team to which the case will be assigned."
    )

    # Add the confidence percentage slider to the interface
    confidence_slider.render()

    # Dropdown listing every case (row) from the CSV.
    case_dropdown = gr.Dropdown(
        choices=case_ids,
        value=None,
        label="Case Id",
    )

    # One input component per schema field.
    inputs = [build_input(field_name) for field_name in field_names]

    # The actual team from the CSV, shown for comparison against the prediction.
    expected_team = gr.Textbox(label="Expected Assigned Team", interactive=False)

    predict_btn = gr.Button("Predict", variant="primary")
    output = gr.Label(label="Predicted Team")
    comparison = gr.HTML(label="Comparison")

    # When a case is selected, populate all the field inputs and the expected team.
    case_dropdown.change(
        fn=load_case,
        inputs=case_dropdown,
        outputs=inputs + [expected_team],
    )

    # Run the prediction using the current field values and compare it.
    predict_btn.click(
        fn=predict,
        inputs=inputs + [expected_team] + [confidence_slider],
        outputs=[output, comparison],
    )


# The section below will run the application
if __name__ == "__main__":
    demo.launch(
        server_name='0.0.0.0',
        server_port=port,
    )
