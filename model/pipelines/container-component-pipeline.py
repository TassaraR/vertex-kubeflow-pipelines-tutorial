import os
from kfp.dsl import (
    container_component,
    ContainerSpec,
    Input,
    Output,
    Dataset,
    Artifact,
    Model,
    pipeline,
)

PIPELINE_NAME = "containerized-penguins-pipeline"
BASE_BUCKET = "gs://kfp-tutorial-b0e2a25a"
PIPELINE_ROOT = os.path.join(BASE_BUCKET, PIPELINE_NAME)


@container_component
def preprocessing_component(
    project_id: str,
    input_path: str,
    num_cols: str,
    cat_cols: str,
    label_col: str,
    output_train_path: Output[Dataset],
    output_test_path: Output[Dataset],
    output_encoder_path: Output[Artifact],
):
    return ContainerSpec(
        image=f"us-central1-docker.pkg.dev/{project_id}/pipeline-tutorial/preprocessing:latest",  # noqa: E501
        command=["python", "runner.py"],
        args=[
            "--input-path",
            input_path,
            "--num-cols",
            num_cols,
            "--cat-cols",
            cat_cols,
            "--label-col",
            label_col,
            "--output-train-path",
            output_train_path.uri,
            "--output-test-path",
            output_test_path.uri,
            "--output-encoder-path",
            output_encoder_path.uri,
        ],
    )


@container_component
def training_component(
    project_id: str,
    num_cols: str,
    cat_cols: str,
    label_col: str,
    input_train: Input[Dataset],
    input_test: Input[Dataset],
    encoder_path: Input[Artifact],
    output_model: Output[Model],
):
    return ContainerSpec(
        image=f"us-central1-docker.pkg.dev/{project_id}/pipeline-tutorial/training:latest",
        command=["python", "runner.py"],
        args=[
            "--input-train",
            input_train.uri,
            "--input-test",
            input_test.uri,
            "--num-cols",
            num_cols,
            "--cat-cols",
            cat_cols,
            "--label-col",
            label_col,
            "--encoder-path",
            encoder_path.uri,
            "--output-model",
            output_model.uri,
        ],
    )


@pipeline(
    name=PIPELINE_NAME,
    description="Penguins tutorial pipeline",
    pipeline_root=PIPELINE_ROOT,
)
def penguins_pipeline(
    project_id: str,
    input_path: str,
    num_cols: str,
    cat_cols: str,
    label_col: str,
):
    preproc_step = preprocessing_component(
        project_id=project_id,
        input_path=input_path,
        num_cols=num_cols,
        cat_cols=cat_cols,
        label_col=label_col,
    )

    train_step = training_component(  # noqa: F841
        project_id=project_id,
        input_train=preproc_step.outputs["output_train_path"],
        input_test=preproc_step.outputs["output_test_path"],
        encoder_path=preproc_step.outputs["output_encoder_path"],
        num_cols=num_cols,
        cat_cols=cat_cols,
        label_col=label_col,
    )
