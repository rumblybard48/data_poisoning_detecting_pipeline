import numpy as np


IGNORE_ZERO_DETECTORS = {
    "duplicate",
    "near_duplicate",
}


def normalize_scores(scores):
    """
    Normalize detector scores to the range [0, 1].

    NaN and infinite values are treated as invalid and
    are excluded from the normalization.
    """

    scores = np.asarray(scores, dtype=np.float64)

    valid = np.isfinite(scores)

    if not np.any(valid):
        return np.full(
            len(scores),
            np.nan,
            dtype=np.float64
        )

    minimum = np.min(scores[valid])
    maximum = np.max(scores[valid])


    if maximum == minimum:

        normalized = np.zeros(
            len(scores),
            dtype=np.float64
        )

        normalized[~valid] = np.nan

        return normalized

    normalized = np.full(
        len(scores),
        np.nan,
        dtype=np.float64
    )

    normalized[valid] = (
        scores[valid] - minimum
    ) / (
        maximum - minimum
    )

    return normalized


def combine_scores(scores):
    """
    Combine scores from multiple poisoning detectors.

    Parameters
    ----------
    scores : dict

        Example:

        {
            "statistical": statistical_scores,
            "label_consistency": label_scores,
            "isolation_forest": isolation_scores,
            "duplicate": duplicate_scores,
            "near_duplicate": near_duplicate_scores,
            "knn": knn_scores
        }

    Returns
    -------
    numpy.ndarray

        One combined score for every row.

    Behaviour
    ---------
    Duplicate and near-duplicate detectors commonly
    return 0 when they find no duplicate evidence.

    Those zeroes are therefore ignored when calculating
    the average.

    Zeroes from the other detectors remain valid scores.
    """

    print("\nCombining detector scores...")

    if not scores:
        raise ValueError(
            "No detector scores were provided."
        )

    normalized_scores = {}

    detector_names = list(scores.keys())

    expected_length = None



    for name, detector_scores in scores.items():

        print(f"Normalizing: {name}")

        detector_scores = np.asarray(
            detector_scores,
            dtype=np.float64
        )

        if expected_length is None:

            expected_length = len(
                detector_scores
            )

        elif len(detector_scores) != expected_length:

            raise ValueError(
                f"Detector '{name}' has "
                f"{len(detector_scores)} scores, "
                f"but expected "
                f"{expected_length}."
            )

        normalized_scores[name] = normalize_scores(
            detector_scores
        )


    detector_matrix = np.column_stack(
        [
            normalized_scores[name]
            for name in detector_names
        ]
    )



    valid_matrix = np.isfinite(
        detector_matrix
    )

    for detector_index, name in enumerate(
        detector_names
    ):


        if name in IGNORE_ZERO_DETECTORS:

            zero_mask = (
                detector_matrix[:, detector_index]
                == 0
            )

            valid_matrix[
                zero_mask,
                detector_index
            ] = False



    combined_scores = np.zeros(
        expected_length,
        dtype=np.float64
    )


    contributing_detectors = np.sum(
        valid_matrix,
        axis=1
    )

    for row_index in range(
        expected_length
    ):

        valid_scores = detector_matrix[
            row_index,
            valid_matrix[row_index]
        ]

        if len(valid_scores) > 0:

            combined_scores[row_index] = (
                np.mean(valid_scores)
            )

        else:

            combined_scores[row_index] = 0.0


    print(
        f"Detectors used: "
        f"{len(detector_names)}"
    )

    print(
        f"Rows: "
        f"{expected_length}"
    )

    print(
        "\nDetector contribution policy:"
    )

    for name in detector_names:

        if name in IGNORE_ZERO_DETECTORS:

            print(
                f"  {name}: "
                f"zero = no evidence "
                f"(ignored)"
            )

        else:

            print(
                f"  {name}: "
                f"zero = valid score"
            )



    print(
        "\nDetector contribution statistics:"
    )

    unique_counts, count_frequencies = np.unique(
        contributing_detectors,
        return_counts=True
    )

    for count, frequency in zip(
        unique_counts,
        count_frequencies
    ):

        print(
            f"  {int(count)} detector(s): "
            f"{int(frequency):,} rows"
        )



    print(
        "\nCombined score statistics:"
    )

    print(
        f"  Minimum: "
        f"{np.min(combined_scores):.6f}"
    )

    print(
        f"  Maximum: "
        f"{np.max(combined_scores):.6f}"
    )

    print(
        f"  Mean:    "
        f"{np.mean(combined_scores):.6f}"
    )

    print(
        f"  Median:  "
        f"{np.median(combined_scores):.6f}"
    )

    return combined_scores