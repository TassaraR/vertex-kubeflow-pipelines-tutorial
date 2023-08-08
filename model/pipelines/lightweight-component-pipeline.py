import os
from kfp.dsl import (
    component,
    Input,
    Output,
    Dataset,
    Artifact,
    Model,
    Metrics,
    ClassificationMetrics,
    pipeline,
)


PIPELINE_NAME = "lightweight-penguins-pipeline"
BASE_BUCKET = "gs://kfp-tutorial-b0e2a25a"
PIPELINE_ROOT = os.path.join(BASE_BUCKET, PIPELINE_NAME)

REPO = "us-central1-docker.pkg.dev"
PROJECT = "tassarar-ml"
BASE_PATH = "pipeline-tutorial"
IMG_BASE_PATH = os.path.join(REPO, PROJECT, BASE_PATH)


@component(base_image=os.path.join(IMG_BASE_PATH, "preprocessing:latest"))
def preprocessing_component(
    input_path: str,
    num_cols: str,
    cat_cols: str,
    label_col: str,
    output_train_path: Output[Dataset],
    output_test_path: Output[Dataset],
    output_encoder_path: Output[Artifact],
):
    import joblib
    import pandas as pd
    from prepare_data import PreprocessData

    input_data = pd.read_csv(input_path)
    parse_cat_cols = cat_cols.split(",")
    parse_num_cols = num_cols.split(",")

    preproc = PreprocessData(
        input_data=input_data,
        categorical_cols=parse_cat_cols,
        numerical_cols=parse_num_cols,
        label_col=label_col,
    )
    preproc.build()

    ds_train, y_train = preproc.get_train_set()
    ds_train[label_col] = y_train
    ds_train.to_csv(output_train_path.path, index=False)

    ds_test, y_test = preproc.get_test_set()
    ds_test[label_col] = y_test
    ds_test.to_csv(output_test_path.path, index=False)

    joblib.dump(preproc.get_encoder(), output_encoder_path.path)

    output_train_path.metadata["shape"] = str(ds_train.shape)
    output_train_path.metadata["columns"] = ds_train.columns.tolist()
    output_test_path.metadata["shape"] = str(ds_test.shape)
    output_test_path.metadata["columns"] = ds_test.columns.tolist()
    output_encoder_path.metadata["params"] = str(preproc.get_encoder().classes_)


@component(base_image=os.path.join(IMG_BASE_PATH, "training:latest"))
def training_component(
    num_cols: str,
    cat_cols: str,
    label_col: str,
    input_train: Input[Dataset],
    input_test: Input[Dataset],
    encoder_path: Input[Artifact],
    output_model: Output[Model],
    eval_metrics: Output[Metrics],
    confusion_matrix: Output[ClassificationMetrics],
):
    import joblib
    import pandas as pd
    from trainer import create_pipeline, evaluate_model

    encoder = joblib.load(encoder_path.path)

    train = pd.read_csv(input_train.path)
    train_set = train.drop(columns=[label_col])
    train_label = train[label_col].to_numpy()

    test = pd.read_csv(input_test.path)
    test_set = test.drop(columns=[label_col])
    test_label = test[label_col].to_numpy()

    parse_cat_cols = cat_cols.split(",")
    parse_num_cols = num_cols.split(",")

    clf = create_pipeline(categorical_cols=parse_cat_cols, numeric_cols=parse_num_cols)

    clf.fit(train_set, train_label)

    metrics = evaluate_model(
        model=clf, eval_set=test_set, eval_y=test_label, encoder=encoder
    )

    joblib.dump(clf, output_model.path)

    # Kubeflow artifact's metrics & metadata
    output_model.metadata["params"] = str(clf.get_params())

    eval_metrics.log_metric("accuracy", metrics["metrics"]["accuracy"])
    for model_class, class_metrics in metrics["metrics"].items():
        if isinstance(class_metrics, dict):
            for metric, score in class_metrics.items():
                eval_metrics.log_metric(f"{model_class} - {metric}", score)

    conf_matrix = metrics["confusion-matrix"]
    conf_matrix_labels = [lbl.split("_")[1] for lbl in conf_matrix.index]

    confusion_matrix.log_confusion_matrix(
        conf_matrix_labels, conf_matrix.T.to_numpy().tolist()
    )


@pipeline(
    name=PIPELINE_NAME,
    description="Penguins tutorial pipeline",
    pipeline_root=PIPELINE_ROOT,
)
def penguins_pipeline(
    input_path: str,
    num_cols: str,
    cat_cols: str,
    label_col: str,
):

    preproc_step = preprocessing_component(
        input_path=input_path,
        num_cols=num_cols,
        cat_cols=cat_cols,
        label_col=label_col,
    )

    train_step = training_component(  # noqa: F841
        input_train=preproc_step.outputs["output_train_path"],
        input_test=preproc_step.outputs["output_test_path"],
        encoder_path=preproc_step.outputs["output_encoder_path"],
        num_cols=num_cols,
        cat_cols=cat_cols,
        label_col=label_col,
    )
