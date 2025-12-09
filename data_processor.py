import os
import re
import logging
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Cleans, normalizes, and merges all datasets from DataLoader into
    unified ML-ready training datasets.
    """

    def __init__(self, data_dir="data"):
        self.root = Path(data_dir)
        self.raw = self.root / "raw"
        self.processed = self.root / "processed"
        self.processed.mkdir(exist_ok=True)

    # ----------------------------------------------------------
    # Utility cleaners
    # ----------------------------------------------------------
    @staticmethod
    def clean_text(x):
        if not isinstance(x, str):
            return ""
        x = re.sub(r"\s+", " ", x)
        x = x.strip()
        return x

    @staticmethod
    def extract_company_domain(url):
        if not isinstance(url, str) or url.strip() == "":
            return ""
        m = re.search(r"([A-Za-z0-9-]+\.(com|in|tech|org|biz|net))", url)
        return m.group(1) if m else ""

    # ----------------------------------------------------------
    # Expected output schema for job dataset
    # ----------------------------------------------------------
    def normalize_job_df(self, df, label_col="label", text_col="description"):
        df = df.copy()

        # Ensure required columns exist
        df = df.rename(columns={text_col: "job_description"})
        
        # Remove empty descriptions
        df = df[df["job_description"].notna()]
        df = df[df["job_description"].str.strip() != ""]

        # CLEAN LABELS (fixes NaN problem)
        df = df[df[label_col].notna()]
        df[label_col] = df[label_col].astype(int)

        return df[["job_description", label_col]]

    # ----------------------------------------------------------
    # Load raw datasets and process them
    # ----------------------------------------------------------
    def load_real_or_fake(self):
        path = self.raw / "real_or_fake"
        files = list(path.glob("*.csv")) + list(path.glob("*.tsv"))
        if not files:
            logger.warning("Real-or-fake dataset missing.")
            return pd.DataFrame()

        df = pd.read_csv(files[0])
        # Kaggle dataset uses 'fraudulent' = 1 or 0
        df["label"] = df["fraudulent"]

        return self.normalize_job_df(df, "label", None)

    def load_fake_jobs(self):
        path = self.raw / "fake_jobs"
        files = list(path.glob("*.csv"))
        if not files:
            logger.warning("fake_jobs dataset missing.")
            return pd.DataFrame()

        df = pd.read_csv(files[0])
        return self.normalize_job_df(df, "label", 1)

    def load_indeed_jobs(self):
        path = self.raw / "indeed_jobs"
        files = list(path.glob("*.csv"))
        if not files:
            logger.warning("Indeed jobs dataset missing.")
            return pd.DataFrame()

        df = pd.read_csv(files[0])
        return self.normalize_job_df(df, "label", 0)

    # ----------------------------------------------------------
    # Process phishing emails dataset
    # ----------------------------------------------------------
    def load_phishing_emails(self):
        path = self.raw / "phishing_emails"
        if not path.exists():
            logger.warning("Phishing emails dataset missing.")
            return pd.DataFrame()

        try:
            from datasets import load_from_disk
            ds = load_from_disk(str(path))
            df = ds.to_pandas()
        except Exception as e:
            logger.error(f"Failed loading phishing emails: {e}")
            return pd.DataFrame()

        df.rename(
            columns={
                "subject": "email_subject",
                "body": "email_body",
                "label": "email_label",
            },
            inplace=True,
        )

        df["email_subject"] = df["email_subject"].astype(str).apply(self.clean_text)
        df["email_body"] = df["email_body"].astype(str).apply(self.clean_text)

        # Assume dataset already uses 1=phishing, 0=legit
        return df[["email_subject", "email_body", "email_label"]]

    # ----------------------------------------------------------
    # MASTER PROCESSOR
    # ----------------------------------------------------------
    def build_job_training_dataset(self):
        logger.info("Processing job datasets...")

        # --------------------------
        # Load 3 datasets safely
        # --------------------------
        df_real_or_fake = self.load_real_or_fake()        # contains 0 + 1
        df_fake_only = self.load_fake_jobs()              # contains 1
        df_indeed = self.load_indeed_jobs()               # contains real jobs (0)

        frames = []

        if df_real_or_fake is not None and not df_real_or_fake.empty:
            frames.append(df_real_or_fake)

        if df_fake_only is not None and not df_fake_only.empty:
            frames.append(df_fake_only)

        if df_indeed is not None and not df_indeed.empty:
            df_indeed["label"] = 0
            frames.append(df_indeed)

        if not frames:
            raise ValueError("No job datasets loaded. Cannot continue.")

        # --------------------------
        # Combine datasets
        # --------------------------
        df = pd.concat(frames, ignore_index=True)

        # --------------------------
        # Clean text
        # --------------------------
        df["job_description"] = df["job_description"].fillna("").astype(str)
        df["label"] = df["label"].astype(int)

        # --------------------------
        # Drop empty rows
        # --------------------------
        df = df[df["job_description"].str.strip() != ""]

        # --------------------------
        # Remove duplicates
        # --------------------------
        df = df.drop_duplicates(subset=["job_description"])

        # --------------------------
        # Save
        # --------------------------
        out_path = Path("data/processed/jobs.parquet")
        df.to_parquet(out_path, index=False)

        # --------------------------
        # Print sanity check
        # --------------------------
        logger.info("Final dataset size: %s", df.shape)
        logger.info("Label distribution:\n%s", df["label"].value_counts())

        return df

    def build_email_training_dataset(self):
        logger.info("Processing email datasets...")
        df = self.load_phishing_emails()
        if df.empty:
            logger.warning("No phishing emails available.")
        else:
            save_path = self.processed / "emails.parquet"
            df.to_parquet(save_path, index=False)
            logger.info(f"Saved processed email dataset → {save_path}")
        return df


def main():
    p = DataProcessor()
    p.build_job_training_dataset()
    p.build_email_training_dataset()
    print("Processing complete.")


if __name__ == "__main__":
    main()
