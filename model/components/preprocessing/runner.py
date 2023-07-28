import argparse
import pickle
import pandas as pd
import gcsfs
import google.auth
from prepare_data import PreprocessData


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", required=True, help="*.csv")
    parser.add_argument("--num-cols", required=True, help="col1,col2,col3")
    parser.add_argument("--cat-cols", required=True, help="col4,col5,col6")
    parser.add_argument("--label-col", required=True, help="col7")
    parser.add_argument("--output-train-path", required=True, help="*.csv")
    parser.add_argument("--output-test-path", required=True, help="*.csv")
    parser.add_argument("--output-encoder-path", required=True, help="*.pkl")

    args = parser.parse_args()

    # Ingestion
    input_data = pd.read_csv(args.input_path)
    parse_cat_cols = args.cat_cols.split(",")
    parse_num_cols = args.num_cols.split(",")

    preproc = PreprocessData(
        input_data=input_data,
        categorical_cols=parse_cat_cols,
        numerical_cols=parse_num_cols,
        label_col=args.label_col,
    )
    preproc.build()

    ds_train, y_train = preproc.get_train_set()
    ds_train[args.label_col] = y_train
    ds_train.to_csv(args.output_train_path, index=False)

    ds_test, y_test = preproc.get_test_set()
    ds_test[args.label_col] = y_test
    ds_test.to_csv(args.output_test_path, index=False)

    if re.findall(r"\/gcs\/.*|gs:\/\/.*", args.output_encoder_path):
        credentials, project_id = google.auth.default()
        fs = gcsfs.GCSFileSystem(project=project_id, token=credentials)
        with fs.open(args.output_encoder_path, "wb") as f:
            pickle.dump(preproc.get_encoder(), f)
    else:
        with open(args.output_encoder_path, "wb") as f:
            pickle.dump(preproc.get_encoder(), f)
