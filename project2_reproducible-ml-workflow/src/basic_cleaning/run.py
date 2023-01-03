#!/usr/bin/env python
"""
Performs basic cleaning on the data and save the results in Weights & Biases
"""
import argparse
import logging
import wandb
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    artifact_local_path = run.use_artifact(args.input_artifact).file()

    ######################
    # YOUR CODE HERE     #
    ######################
    logger.info("Download artifact: %s", args.input_artifact)
    df = pd.read_csv(artifact_local_path)

    # Cleaning steps
    # Drop outliers
    logger.info(
        f"Cleaning step 1: Retain items with price value between {args.min_price} and {args.max_price} dollars")
    min_price = args.min_price
    max_price = args.max_price
    idx = df["price"].between(min_price, max_price)
    df = df[idx].copy()

    # Convert last_review to datetime
    logger.info("Cleaning step 2: Convert last_review to datetime")
    df['last_review'] = pd.to_datetime(df['last_review'])

    # Latitude/Longitude boundaries
    logger.info("Cleaning step 3: Limiting longitude and latitude to NYC area")
    idx = df['longitude'].between(-74.25, -
                                  73.50) & df['latitude'].between(40.5, 41.2)
    df = df[idx].copy()

    # Save file as CSV and load W&B artifact
    logger.info(f"Saving clean_sample.csv in local directory")
    df.to_csv("clean_sample.csv", index=False)

    # Uploading artifact to W&B
    logger.info(f"Uploading {args.output_artifact} to Weights & Biases")
    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description
    )
    artifact.add_file("clean_sample.csv")
    run.log_artifact(artifact)

    artifact.wait()
    logger.info("Artifact uploaded")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="This steps cleans the data")

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Name of the input artifact contained in W&B",
        required=True
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="Name of the output artifact to be uploaded in W&B",
        required=True
    )

    parser.add_argument(
        "--output_type",
        type=str,
        help="The type for the output artifact",
        required=True
    )

    parser.add_argument(
        "--output_description",
        type=str,
        help="A description for the output artifact",
        required=True
    )

    parser.add_argument(
        "--min_price",
        type=float,
        help="A description for the minimum price",
        required=True
    )

    parser.add_argument(
        "--max_price",
        type=float,
        help="A description for the maximum price",
        required=True
    )

    args = parser.parse_args()

    go(args)
