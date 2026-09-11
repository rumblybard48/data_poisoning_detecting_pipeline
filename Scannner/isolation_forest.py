"""
Isolation Forest anomaly detector.

Detects globally unusual observations in multidimensional feature space.

This module:
    - accepts a pandas DataFrame
    - selects numeric features
    - handles NaN and infinite values
    - trains an Isolation Forest
    - produces one anomaly score per input row

IMPORTANT:
    This module does NOT determine whether a sample is poisoned.
    It only produces anomaly scores.

Score interpretation:
    Higher score = more anomalous
    Lower score = more normal
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


class IsolationForestDetector:
    """
    Isolation Forest-based anomaly detector.

    Parameters
    ----------
    n_estimators : int
        Number of isolation trees.

    max_samples : int, "auto", or float
        Number of samples used to build each tree.

        For a large dataset such as EMBER, this should be kept
        considerably smaller than the total number of observations.

    contamination : float or "auto"
        Expected proportion of anomalies.

        "auto" is recommended because this module only produces
        anomaly scores and does not make the final poisoning decision.

    random_state : int
        Random seed for reproducibility.

    n_jobs : int
        Number of CPU cores to use. -1 uses all available cores.
    """

    def __init__(
        self,
        n_estimators=100,
        max_samples=10000,
        contamination="auto",
        random_state=42,
        n_jobs=-1,
    ):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.model = None
        self.feature_columns = None

    def _prepare_features(self, df):
        """
        Select numeric features and handle invalid values.

        Returns
        -------
        numpy.ndarray
            Numeric feature matrix in float32 format.
        """

        if not isinstance(df, pd.DataFrame):
            raise TypeError("Expected a pandas DataFrame.")


        numeric_df = df.select_dtypes(include=[np.number])

        if numeric_df.empty:
            raise ValueError("No numeric features found.")

        self.feature_columns = numeric_df.columns.tolist()

        print(f"Numeric features selected: {len(self.feature_columns):,}")


        X = numeric_df.to_numpy(dtype=np.float32, copy=True)

 
        invalid_mask = ~np.isfinite(X)

        if invalid_mask.any():

            invalid_count = int(invalid_mask.sum())

            print(
                f"Warning: {invalid_count:,} invalid values found."
            )


            valid_values = np.where(
                np.isfinite(X),
                X,
                np.nan
            )

            medians = np.nanmedian(
                valid_values,
                axis=0
            )

  
            medians = np.nan_to_num(
                medians,
                nan=0.0,
                posinf=0.0,
                neginf=0.0
            )

            rows, columns = np.where(invalid_mask)

            X[rows, columns] = medians[columns]

            print("Invalid values replaced with column medians.")

        else:
            print("No NaN or infinite values found.")

        return X

    def fit(self, df):
        """
        Train the Isolation Forest on the supplied dataset.

        Returns
        -------
        IsolationForestDetector
            Fitted detector.
        """

        X = self._prepare_features(df)

        n_samples = X.shape[0]
        n_features = X.shape[1]


        if isinstance(self.max_samples, int):
            actual_max_samples = min(
                self.max_samples,
                n_samples
            )
        else:
            actual_max_samples = self.max_samples

        print("\nIsolation Forest configuration")
        print("--------------------------------")
        print(f"Samples:      {n_samples:,}")
        print(f"Features:     {n_features:,}")
        print(f"Trees:        {self.n_estimators}")
        print(f"Max samples:  {actual_max_samples}")
        print(f"Contamination:{self.contamination}")
        print(f"CPU workers:  {self.n_jobs}")

        print("\nTraining Isolation Forest...")

        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=actual_max_samples,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )

        self.model.fit(X)

        print("Isolation Forest training complete.")

        return self

    def score_samples(self, df):
        """
        Calculate an anomaly score for every row.

        Higher score = more anomalous.
        Lower score = more normal.

        Returns
        -------
        numpy.ndarray
            One anomaly score per input row.
        """

        if self.model is None:
            raise RuntimeError(
                "Isolation Forest has not been fitted yet."
            )

        X = self._prepare_features(df)

        print("\nCalculating anomaly scores...")


        scores = -self.model.score_samples(X)

        print("Anomaly scoring complete.")

        return scores

    def fit_score(self, df):
        """
        Train the Isolation Forest and calculate anomaly
        scores for every row.
        """

        self.fit(df)

        return self.score_samples(df)