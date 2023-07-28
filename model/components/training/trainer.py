import pandas as pd
import numpy as np
from typing import Union, Optional, List
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


def create_pipeline(categorical_cols: List[str], numeric_cols: List[str]) -> Pipeline:
    column_trans = ColumnTransformer(
        [
            ("scaler", StandardScaler(), numeric_cols),
            ("ohe", OneHotEncoder(), categorical_cols),
        ]
    )

    estimators = [("preprocessing", column_trans), ("model", DecisionTreeClassifier())]

    pipeline = Pipeline(estimators)

    return pipeline


def evaluate_model(
    model: Union[Pipeline, DecisionTreeClassifier],
    eval_set: pd.DataFrame,
    eval_y: np.ndarray,
    encoder: Optional[LabelEncoder] = None,
) -> dict:
    predictions = model.predict(eval_set)

    conf_mat = confusion_matrix(eval_y, predictions)
    conf_mat = pd.DataFrame(conf_mat)

    if encoder:
        target_name = encoder.classes_
        index = [f"true_{x.lower()}" for x in target_name]
        columns = [f"pred_{x.lower()}" for x in target_name]
        conf_mat.index = index
        conf_mat.columns = columns
    else:
        target_name = None

    classification_metrics = classification_report(
        y_true=eval_y, y_pred=predictions, output_dict=True, target_names=target_name
    )
    classification_metrics["accuracy"] = accuracy_score(eval_y, predictions)

    return {"metrics": classification_metrics, "confusion-matrix": conf_mat}
