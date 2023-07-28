import re
import argparse
import pickle
import gcsfs
import google.auth
import pandas as pd
from trainer import create_pipeline, evaluate_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-train", required=True, help="*.csv")
    parser.add_argument("--input-test", required=True, help="*.csv")
    parser.add_argument("--num-cols", required=True, help="col1,col2,col3")
    parser.add_argument("--cat-cols", required=True, help="col4,col5,col6")
    parser.add_argument("--label-col", required=True, help="col7")
    parser.add_argument("--encoder-path", required=True, help="*.pkl")
    parser.add_argument("--output-model", required=True, help="*.pkl")

    args = parser.parse_args()

    train = pd.read_csv(args.input_train)
    train_set = train.drop(columns=[args.label_col])
    train_label = train[args.label_col].to_numpy()

    test = pd.read_csv(args.input_test)
    test_set = test.drop(columns=[args.label_col])
    test_label = test[args.label_col].to_numpy()

    if re.findall(r"\/gcs\/.*|gs:\/\/.*", args.encoder_path):
        credentials, project_id = google.auth.default()
        fs = gcsfs.GCSFileSystem(project=project_id, token=credentials)
        with fs.open(args.encoder_path, "rb") as f:
            encoder = pickle.load(f)
    else:
        with open(args.encoder_path, "rb") as f:
            encoder = pickle.load(f)

    parse_cat_cols = args.cat_cols.split(",")
    parse_num_cols = args.num_cols.split(",")

    clf = create_pipeline(categorical_cols=parse_cat_cols, numeric_cols=parse_num_cols)

    clf.fit(train_set, train_label)

    metrics = evaluate_model(
        model=clf, eval_set=test_set, eval_y=test_label, encoder=encoder
    )

    if re.findall(r"\/gcs\/.*|gs:\/\/.*", args.output_model):
        credentials, project_id = google.auth.default()
        fs = gcsfs.GCSFileSystem(project=project_id, token=credentials)
        with fs.open(args.output_model, "wb") as f:
            pickle.dump(args.output_model, f)
    else:
        with open(args.output_model, "wb") as f:
            pickle.dump(clf, f)
