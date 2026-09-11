import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report

def detect(df):

    print("Running label consistency detection...")

    X = df.drop(columns=["Label"])
    y = df["Label"]

    print(f"Features: {X.shape[1]}")
    print(f"Rows: {X.shape[0]}")

    true_label_probabilities = np.zeros(len(df))
    predictions = np.empty(len(df), dtype=y.dtype)

    skf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    for fold, (train_index, test_index) in enumerate(
        skf.split(X, y),
        start=1
    ):

        print(f"Training fold {fold}/5...")

        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]

        y_train = y.iloc[train_index]

        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train, y_train)

        probabilities = model.predict_proba(X_test)
        predictions[test_index] = model.predict(X_test)

        class_indices = {
            label: index
            for index, label in enumerate(model.classes_)
        }

        for i, row_index in enumerate(test_index):

            actual_label = y.iloc[row_index]

            true_label_probabilities[row_index] = probabilities[
                i,
                class_indices[actual_label]
            ]

    scores = 1.0 - true_label_probabilities
    print("\nAverage consistency score by label:")

    for label in sorted(y.unique()):

        label_scores = scores[
            y.to_numpy() == label
        ]

        print(
            f"Label {label}: "
            f"mean={label_scores.mean():.4f}, "
            f"median={np.median(label_scores):.4f}"
        )
    accuracy = np.mean(
        predictions == y.to_numpy()
    )

    print("\nConfusion matrix:")

    print(
        confusion_matrix(
            y,
            predictions,
            labels=[-1.0, 0.0, 1.0]
        )
    )

    print("\nClassification report:")

    print(
        classification_report(
            y,
            predictions,
            labels=[-1.0, 0.0, 1.0],
            zero_division=0
        )
    )

    print(f"\nOut-of-fold accuracy: {accuracy:.4f}")

    print("\nPrediction distribution:")
    print(
        np.unique(
            predictions,
            return_counts=True
        )
    )
    print("\nLabel consistency detection complete.")

    print("\nTrue-label probability statistics:")
    print("Minimum:", true_label_probabilities.min())
    print("Maximum:", true_label_probabilities.max())
    print("Mean:", true_label_probabilities.mean())
    print("Median:", np.median(true_label_probabilities))

    print("\nLabel consistency score statistics:")
    print("Minimum:", scores.min())
    print("Maximum:", scores.max())
    print("Mean:", scores.mean())
    print("Median:", np.median(scores))

    print("\nFirst 20 label consistency scores:")
    print(scores[:20])

    print("\nLabel distribution:")
    print("\nFeature variation diagnostic:")

    constant_features = []

    for column in X.columns:

        unique_count = X[column].nunique()

        if unique_count <= 1:
            constant_features.append(column)

    print("Constant features:", len(constant_features))

    if constant_features:
        print("First 20 constant features:")
        print(constant_features[:20])


    print("\nLabel variation:")

    print("Unique labels:", y.nunique())
    print("Label values:", y.unique())


    print("\nLabel counts:")

    print(y.value_counts())


    print("\nFeature statistics:")

    print(X.iloc[:, :20].describe().T[[
        "count",
        "mean",
        "std",
        "min",
        "max"
    ]])





    return scores