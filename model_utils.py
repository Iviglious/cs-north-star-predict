import csv
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier


DATA_DIR = Path("data")

EXPECTED_COLUMNS = [
    "case_id",
    "snapshot_at",
    "created_at",
    "channel",
    "case_type",
    "category",
    "subcategory",
    "priority",
    "sla_target_hours",
    "first_response_time_hours",
    "resolution_time_hours",
    "status",
    "resolution_code",
    "escalated",
    "assigned_team",
    "escalation_team",
    "customer_tenure_months",
    "plan_tier",
    "region_uk",
    "age_band",
    "gender",
    "case_summary",
    "sentiment",
    "csat_score",
    "tags",
]
#split features
ROUTING_FEATURES = ["channel", "priority", "sla_target_hours", "plan_tier", "sentiment", "tags"]
CATEGORICAL_ROUTING_FEATURES = ["channel", "priority", "plan_tier", "sentiment"]
NUMERIC_ROUTING_FEATURES = ["sla_target_hours"]
TAG_FEATURE = "tags"
ETHICS_ANALYSIS_COLUMNS = ["customer_tenure_months", "region_uk", "age_band", "gender"]
FUTURE_TARGETS = ["case_type", "category", "subcategory"]
FUTURE_TEXT_FEATURES = ["case_summary"]

MONTH_ORDER = {
    "Jan": 1,
    "Feb": 2,
    "Mar": 3,
    "Apr": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "Aug": 8,
    "Sep": 9,
    "Oct": 10,
    "Nov": 11,
    "Dec": 12,
}
TRAIN_MONTHS = range(1, 7)
TEST_MONTHS = range(7, 10)
HOLDOUT_MONTHS = range(10, 13)
FINAL_TRAIN_MONTHS = range(1, 10)
DEMO_MONTHS = range(10, 13)
PRIMARY_EVALUATION_SLICE = "solved_only"

#data repair functions
def repair_row(row, source_file, row_number):
    repair = None

    if len(row) == len(EXPECTED_COLUMNS) - 1:
        #Assume only missing Gender as per initial Data investigation, for productionised code would want additional validation
        row = row[:20] + [""] + row[20:]
        repair = {
            "source_file": source_file,
            "row_number": row_number,
            "issue": "missing_gender_column",
            "action": "inserted_blank_gender_before_case_summary",
        }

    if len(row) != len(EXPECTED_COLUMNS):
        raise ValueError(
            f"{source_file} row {row_number} has {len(row)} columns after repair; "
            f"expected {len(EXPECTED_COLUMNS)}"
        )

    return row, repair


def load_case_data(data_dir=DATA_DIR):
    rows = []
    repairs = []

    for path in sorted(Path(data_dir).glob("*.csv")):
        with path.open(newline="", encoding="utf-8-sig") as file:
            reader = csv.reader(file)
            header = next(reader)

            if header != EXPECTED_COLUMNS:
                raise ValueError(f"Unexpected header in {path.name}: {header}")

            for row_number, row in enumerate(reader, start=2):
                row, repair = repair_row(row, path.name, row_number)
                rows.append(dict(zip(EXPECTED_COLUMNS, row)) | {"source_file": path.name})
                if repair is not None:
                    repairs.append(repair)

    df = pd.DataFrame(rows)

    for column in df.columns:
        if df[column].dtype == "object":
            df[column] = df[column].str.strip()
    df = df.replace("", pd.NA)

    for column in ["snapshot_at", "created_at"]:
        df[column] = pd.to_datetime(df[column], errors="coerce", utc=True)

    numeric_columns = [
        "sla_target_hours",
        "first_response_time_hours",
        "resolution_time_hours",
        "customer_tenure_months",
        "csat_score",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["escalated"] = df["escalated"].str.lower().map({"true": True, "false": False})
    df["tags"] = df["tags"].apply(
        lambda value: [tag.strip() for tag in str(value).split(";") if tag.strip()]
        if pd.notna(value)
        else []
    )

    repair_log = pd.DataFrame(repairs)
    return df, repair_log


class TagBinarizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.classes_ = None
        self.class_to_index_ = None

    def fit(self, X, y=None):
        self.classes_ = sorted({tag for tags in self._to_tag_lists(X) for tag in tags})
        self.class_to_index_ = {tag: index for index, tag in enumerate(self.classes_)}
        return self

    def transform(self, X):
        encoded = np.zeros((len(X), len(self.classes_)), dtype=int)
        for row_index, tags in enumerate(self._to_tag_lists(X)):
            for tag in tags:
                column_index = self.class_to_index_.get(tag)
                if column_index is not None:
                    encoded[row_index, column_index] = 1
        return encoded

    def get_feature_names_out(self, input_features=None):
        return [f"tag__{tag}" for tag in self.classes_]

    def _to_tag_lists(self, X):
        series = X.squeeze()
        return series.apply(lambda value: value if isinstance(value, list) else []).tolist()


def make_model(model_type):
    if model_type == "logistic":
        return LogisticRegression(max_iter=1000, class_weight="balanced")

    if model_type == "decision_tree":
        return DecisionTreeClassifier(
            random_state=42,
            class_weight="balanced",
            max_depth=8,
        )

    if model_type == "random_forest":
        return RandomForestClassifier(
            random_state=42,
            class_weight="balanced",
            n_estimators=300,
            max_depth=None,
        )

    raise ValueError("model_type must be one of: 'logistic', 'decision_tree', 'random_forest'")


def make_pipeline(model_type):
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_ROUTING_FEATURES,
            ),
            (
                "numeric",
                SimpleImputer(strategy="median"),
                NUMERIC_ROUTING_FEATURES,
            ),
            (
                "tags",
                TagBinarizer(),
                [TAG_FEATURE],
            ),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", make_model(model_type)),
        ]
    )


def add_file_month(df):
    df = df.copy()
    df["file_month_name"] = df["source_file"].str.extract(r"Q\d-([A-Za-z]+)\.csv")[0]
    df["file_month"] = df["file_month_name"].map(MONTH_ORDER)

    if df["file_month"].isna().any():
        unknown_files = df.loc[df["file_month"].isna(), "source_file"].unique()
        raise ValueError(f"Could not infer month from source_file values: {unknown_files}")

    return df


def build_temporal_model_frames(labelled_cases, target_column):
    labelled_cases = add_file_month(labelled_cases)

    split_masks = {
        "train_jan_to_jun": labelled_cases["file_month"].isin(TRAIN_MONTHS),
        "test_july_to_sept": labelled_cases["file_month"].isin(TEST_MONTHS),
        "holdout_oct_to_dec": labelled_cases["file_month"].isin(HOLDOUT_MONTHS),
    }

    frames = {}
    for split_name, mask in split_masks.items():
        split_cases = labelled_cases.loc[mask].copy()
        frames[split_name] = {
            "cases": split_cases,
            "X": split_cases[ROUTING_FEATURES].copy(),
            "y": split_cases[target_column].copy(),
        }

    return frames


def train_temporal_classifier(frames, target_name, model_type="logistic", slice_name=None):
    X_train = frames["train_jan_to_jun"]["X"]
    y_train = frames["train_jan_to_jun"]["y"]
    X_test = frames["test_july_to_sept"]["X"]
    y_test = frames["test_july_to_sept"]["y"]
    X_holdout = frames["holdout_oct_to_dec"]["X"]

    if y_train.nunique() < 2:
        raise ValueError(f"{target_name} needs at least two training classes")

    model = make_pipeline(model_type)
    model.fit(X_train, y_train)

    label = f"{target_name} {model_type}"
    if slice_name is not None:
        label = f"{label} ({slice_name})"

    print(f"\n{label}: train Jan-Jun, test July-Sept")
    print(
        f"Training rows: {len(X_train):,}; "
        f"Q3 test rows: {len(X_test):,}; "
        f"Q4 holdout rows: {len(X_holdout):,}"
    )

    test_accuracy = None
    if X_test.empty:
        print("No labelled Q3 rows are available for this target, so July-Sept cannot be used as a test set.")
    else:
        test_predictions = model.predict(X_test)
        test_accuracy = (test_predictions == y_test.to_numpy()).mean()
        print(classification_report(y_test, test_predictions, zero_division=0))

    metrics = {
        "target": target_name,
        "slice": slice_name,
        "model_type": model_type,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "holdout_rows": len(X_holdout),
        "test_accuracy": test_accuracy,
    }

    return model, metrics


def build_assigned_team_frames_by_slice(cases):
    labelled_cases = add_file_month(cases[cases["assigned_team"].notna()].copy())
    train_cases = labelled_cases[
        labelled_cases["status"].eq("solved")
        & labelled_cases["file_month"].isin(TRAIN_MONTHS)
    ].copy()

    evaluation_slices = {
        "solved_only": labelled_cases["status"].eq("solved"),
        "active_only": labelled_cases["status"].ne("solved"),
        "all_cases": labelled_cases["status"].notna(),
    }

    frames_by_slice = {}
    for slice_name, slice_mask in evaluation_slices.items():
        test_cases = labelled_cases[
            slice_mask & labelled_cases["file_month"].isin(TEST_MONTHS)
        ].copy()
        holdout_cases = labelled_cases[
            slice_mask & labelled_cases["file_month"].isin(HOLDOUT_MONTHS)
        ].copy()

        frames_by_slice[slice_name] = {
            "train_jan_to_jun": {
                "cases": train_cases,
                "X": train_cases[ROUTING_FEATURES].copy(),
                "y": train_cases["assigned_team"].copy(),
            },
            "test_july_to_sept": {
                "cases": test_cases,
                "X": test_cases[ROUTING_FEATURES].copy(),
                "y": test_cases["assigned_team"].copy(),
            },
            "holdout_oct_to_dec": {
                "cases": holdout_cases,
                "X": holdout_cases[ROUTING_FEATURES].copy(),
                "y": holdout_cases["assigned_team"].copy(),
            },
        }

    return frames_by_slice


def build_escalation_team_frames(cases):
    escalation_cases = cases[
        cases["escalated"].eq(True) & cases["escalation_team"].notna()
    ].copy()
    return build_temporal_model_frames(escalation_cases, "escalation_team")


def _filter_by_slice(labelled_cases, slice_name):
    if slice_name == "solved_only":
        return labelled_cases[labelled_cases["status"].eq("solved")].copy()
    if slice_name == "active_only":
        return labelled_cases[labelled_cases["status"].ne("solved")].copy()
    if slice_name == "all_cases":
        return labelled_cases.copy()
    raise ValueError("slice_name must be one of: 'solved_only', 'active_only', 'all_cases'")


def final_train_demo_split(cases, slice_name="solved_only", demo_slice_name="all_cases"):
    labelled_cases = add_file_month(cases[cases["assigned_team"].notna()].copy())
    final_train_cases = _filter_by_slice(labelled_cases, slice_name)
    demo_cases = _filter_by_slice(labelled_cases, demo_slice_name)

    final_train_cases = final_train_cases[
        final_train_cases["file_month"].isin(FINAL_TRAIN_MONTHS)
    ].copy()
    demo_cases = demo_cases[demo_cases["file_month"].isin(DEMO_MONTHS)].copy()
    return final_train_cases, demo_cases