from typing import Optional, List
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTENC


class PreprocessData:
    def __init__(
        self,
        input_data: pd.DataFrame,
        categorical_cols: List[str],
        numerical_cols: List[str],
        label_col: str,
        test_pct: float = 0.3,
    ) -> None:
        self.input_data = input_data
        self.categorical_cols = categorical_cols
        self.numerical_cols = numerical_cols
        self.label_col = label_col
        self.test_pct = test_pct

        self._lbl_enc = None
        self._train_set = None
        self._test_set = None

    def _encode_categorical(self) -> None:
        self._lbl_enc = LabelEncoder()
        self._lbl_enc.fit(self.input_data[self.label_col])

    def _apply_encoding(self) -> np.array:
        if not self._lbl_enc:
            self._encode_categorical()
        return self._lbl_enc.transform(self.input_data[self.label_col])

    def get_encoder(self) -> LabelEncoder:
        if not self._lbl_enc:
            raise ValueError("No label encoding found")
        return self._lbl_enc

    def _train_test_split(self) -> None:
        if not self._lbl_enc:
            self._apply_encoding()

        label = self._lbl_enc.transform(self.input_data[self.label_col])
        ds = self.input_data.iloc[:, self.input_data.columns != self.label_col]

        ds_train, ds_test, y_train, y_test = train_test_split(
            ds,
            label,
            test_size=self.test_pct,
            stratify=self.input_data[self.label_col],
            random_state=0,
        )

        self._train_set = (ds_train, y_train)
        self._test_set = (ds_test, y_test)

    def _balance(self, **smote_kwargs) -> None:
        if not self._train_set:
            raise ValueError("Data has not yet been split")

        ds_train, ds_test = self._train_set

        cat_indeces = [
            n for n, x in enumerate(ds_train.columns.isin(self.categorical_cols)) if x
        ]

        smote_params = dict(
            categorical_features=cat_indeces,
            sampling_strategy="all",
            random_state=0,
        )
        if smote_kwargs:
            smote_params = {**smote_params, **smote_kwargs}

        sm = SMOTENC(**smote_params)

        ds_train, y_train = sm.fit_resample(ds_train, ds_test)
        self._train_set = (ds_train, y_train)

    def build(self) -> None:
        # label encoder
        self._apply_encoding()
        # split
        self._train_test_split()
        # upsample
        self._balance()

    def get_train_set(self) -> Optional[pd.DataFrame]:
        return self._train_set

    def get_test_set(self) -> Optional[pd.DataFrame]:
        return self._test_set
