from functools import reduce
import numpy as np
import pandas as pd


"""
Best model checkpoints:

B2:
rgb_qf_input_efficientnet_b2_1593780039

B3:
rgb_qf_input_efficientnet_b3_1592261292
ycbcr_qf_input_efficientnet_b3_1592654065

B4:
rgb_qf_input_efficientnet_b4_1594044310

B5:
rgb_efficientnet_b5_1591917676

B6:

"""


def blend_submissions() -> None:
    submissions_path = "submissions/"

    submissions_to_blend = [
        "rgb_efficientnet_b4_tta_lb_0_927_submission.csv",
        "rgb_efficientnet_b3_lb_0_931_submission.csv",
        "rgb_efficientnet_b3_lb_0_933_submission.csv",
        "rgb_efficientnet_b3_lb_0_933_submission.csv",
        "rgb_efficientnet_b3_lb_0_933_submission.csv",
        "rgb_efficientnet_b3_lb_0_933_submission.csv",
        "rgb_efficientnet_b3_lb_0_922_submission.csv",
        "rgb_efficientnet_b5_lb_0_912_submission.csv",
        "ycbcr_efficientnet_b3_lb_0_929_submission.csv",
        "ycbcr_efficientnet_b3_lb_0_929_submission.csv",
        "ycbcr_efficientnet_b3_lb_0_929_submission.csv",
        "rgb_efficientnet_b6_tta_lb_0_925_submission.csv",
    ]
    best_base_submission = "rgb_efficientnet_b3_lb_0_933_submission.csv"
    weights = [0.2, 0.3, 0.5]

    submission_data_frames = [
        pd.read_csv(f"{submissions_path}{file}").rename(
            columns={"Label": file.strip(".csv")}
        )
        for file in submissions_to_blend
    ]
    best_base_submission_df = pd.read_csv(
        f"{submissions_path}{best_base_submission}"
    )

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

    cutoff_low = 0.1
    cutoff_high = 0.9

    # Stacked mean:
    merged_df["Label"] = merged_df["mean"]
    merged_df[["Id", "Label"]].to_csv(
        f"{submissions_path}stack_mean.csv", index=False
    )

    # Stacked median:
    merged_df["Label"] = merged_df["median"]
    merged_df[["Id", "Label"]].to_csv(
        f"{submissions_path}stack_median.csv", index=False
    )

    # Stacked pushout mean:
    merged_df["Label"] = np.where(
        np.all(merged_df.iloc[:, 1:n_col] > cutoff_low, axis=1),
        merged_df["max"],
        np.where(
            np.all(merged_df.iloc[:, 1:n_col] < cutoff_high, axis=1),
            merged_df["min"],
            merged_df["mean"],
        ),
    )
    merged_df[["Id", "Label"]].to_csv(
        f"{submissions_path}stack_minmax_mean.csv", index=False,
    )

    # Stacked pushout median:
    merged_df["Label"] = np.where(
        np.all(merged_df.iloc[:, 1:n_col] > cutoff_low, axis=1),
        1,
        np.where(
            np.all(merged_df.iloc[:, 1:n_col] < cutoff_high, axis=1),
            0,
            merged_df["median"],
        ),
    )
    merged_df[["Id", "Label"]].to_csv(
        f"{submissions_path}stack_pushout_median.csv", index=False,
    )

    # Stacked minmax median:
    merged_df["Label"] = np.where(
        np.all(merged_df.iloc[:, 1:n_col] > cutoff_low, axis=1),
        merged_df["max"],
        np.where(
            np.all(merged_df.iloc[:, 1:n_col] < cutoff_high, axis=1),
            merged_df["min"],
            merged_df["median"],
        ),
    )
    merged_df[["Id", "Label"]].to_csv(
        f"{submissions_path}stack_minmax_median.csv", index=False,
    )

    # Stacked minmax best base:
    base_labels = best_base_submission_df["Label"]
    merged_df["Label"] = np.where(
        np.all(merged_df.iloc[:, 1:n_col] > cutoff_low, axis=1),
        merged_df["max"],
        np.where(
            np.all(merged_df.iloc[:, 1:n_col] < cutoff_high, axis=1),
            merged_df["min"],
            base_labels,
        ),
    )
    merged_df[["Id", "Label"]].to_csv(
        f"{submissions_path}stack_minmax_best_base.csv", index=False,
    )


if __name__ == "__main__":
    blend_submissions()
