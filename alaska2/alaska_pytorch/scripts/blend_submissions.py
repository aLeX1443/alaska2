from functools import reduce
import numpy as np
import pandas as pd


def blend_submissions() -> None:
    submissions_path = "submissions/"

    submissions_to_blend = [
        # "efficientnet_b0_lb_0_871_submission.csv",
        "efficientnet_b0_lb_0_896_submission.csv",
        "efficientnet_b3_lb_0_900_submission.csv",
        "efficientnet_b3_lb_0_914_submission.csv",
        "efficientnet_b3_lb_0_921_submission.csv",
    ]
    weights = [0.2, 0.3, 0.5]

    submission_data_frames = [
        pd.read_csv(f"{submissions_path}{file}").rename(
            columns={"Label": file.strip(".csv")}
        )
        for file in submissions_to_blend
    ]

    # Use the Id column to merge all the submissions into one table, where each
    # submission's predictions are in a separate column.
    merged_df = reduce(
        lambda left, right: pd.merge(left, right, on="Id"),
        submission_data_frames,
    )

    n_col = len(merged_df.columns)
    merged_df["max"] = merged_df.iloc[:, 1:n_col].max(axis=1)
    merged_df["min"] = merged_df.iloc[:, 1:n_col].min(axis=1)
    merged_df["mean"] = merged_df.iloc[:, 1:n_col].mean(axis=1)
    merged_df["median"] = merged_df.iloc[:, 1:n_col].median(axis=1)

    cutoff_lo = 0.3
    cutoff_hi = 0.7

    # Stacked mean:
    merged_df["Label"] = merged_df["mean"]
    merged_df[["Id", "Label"]].to_csv(
        f"{submissions_path}stack_mean.csv", index=False, float_format="%.6f"
    )

    # Stacked median:
    merged_df["Label"] = merged_df["median"]
    merged_df[["Id", "Label"]].to_csv(
        f"{submissions_path}stack_median.csv", index=False, float_format="%.6f"
    )

    # Stacked pushout mean:
    merged_df["Label"] = np.where(
        np.all(merged_df.iloc[:, 1:n_col] > cutoff_lo, axis=1),
        merged_df["max"],
        np.where(
            np.all(merged_df.iloc[:, 1:n_col] < cutoff_hi, axis=1),
            merged_df["min"],
            merged_df["mean"],
        ),
    )
    merged_df[["Id", "Label"]].to_csv(
        f"{submissions_path}stack_minmax_mean.csv",
        index=False,
        float_format="%.6f",
    )

    # Stacked pushout median:
    merged_df["Label"] = np.where(
        np.all(merged_df.iloc[:, 1:n_col] > cutoff_lo, axis=1),
        1,
        np.where(
            np.all(merged_df.iloc[:, 1:n_col] < cutoff_hi, axis=1),
            0,
            merged_df["median"],
        ),
    )
    merged_df[["Id", "Label"]].to_csv(
        f"{submissions_path}stack_pushout_median.csv",
        index=False,
        float_format="%.6f",
    )

    # Stacked minmax median:
    merged_df["Label"] = np.where(
        np.all(merged_df.iloc[:, 1:n_col] > cutoff_lo, axis=1),
        merged_df["max"],
        np.where(
            np.all(merged_df.iloc[:, 1:n_col] < cutoff_hi, axis=1),
            merged_df["min"],
            merged_df["median"],
        ),
    )
    merged_df[["Id", "Label"]].to_csv(
        f"{submissions_path}stack_minmax_median.csv",
        index=False,
        float_format="%.6f",
    )


if __name__ == "__main__":
    blend_submissions()
