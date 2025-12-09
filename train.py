import os
import logging
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from imblearn.over_sampling import SMOTE
import pickle

from data_loader import DataLoader
from data_processor import DataProcessor

# -------------------------
# Logging Setup
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================
#                  MODEL TRAINER CLASS
# ============================================================
class ModelTrainer:

    def __init__(self, config):
        self.config = config
        self.data_loader = DataLoader()
        self.data_processor = DataProcessor()

        # Paths
        self.root_dir = Path(__file__).parent
        self.models_dir = self.root_dir / "models"
        self.models_dir.mkdir(exist_ok=True)

        embed_name = config.get("model_name", "all-MiniLM-L6-v2")
        logger.info(f"Loading embedding model: {embed_name}")
        self.embedder = SentenceTransformer(embed_name)

    # ------------------------------------------------------------
    def load_data(self):
        """Load dataset and FIX NaN issues."""
        logger.info("Loading processed job dataset...")

        data_path = Path("data/processed/jobs.parquet")

        if not data_path.exists():
            logger.info("Processed dataset missing — rebuilding...")
            self.data_loader.download_all()
            self.data_processor.build_job_training_dataset()

        df = pd.read_parquet(data_path)

        # ----------------------------------------
        # CLEAN DATA (fix NaN label problem)
        # ----------------------------------------
        df = df.dropna(subset=["job_description", "label"])
        df["job_description"] = df["job_description"].astype(str).str.strip()
        df = df[df["job_description"] != ""]

        df["label"] = df["label"].astype(int)

        logger.info(f"Dataset after cleaning: {df.shape}")

        # Train/val split
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])

        return train_df, val_df

    # ------------------------------------------------------------
    def embed_texts(self, texts):
        logger.info(f"Embedding {len(texts)} samples...")
        return self.embedder.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True
        )

    # ------------------------------------------------------------
    def train(self):

        train_df, val_df = self.load_data()

        # ------------------------------------------------------------
        # CHECK LABEL DISTRIBUTION BEFORE TRAINING
        # ------------------------------------------------------------
        logger.info("Label distribution in FULL dataset:")
        logger.info(train_df["label"].value_counts())

        if len(train_df["label"].unique()) < 2:
            raise ValueError(
                "❌ Training set contains ONLY ONE CLASS. "
                "You must include at least one fake-job dataset (fraudulent=1)."
            )

        # ------------------------------------------------------------
        # Embed text
        # ------------------------------------------------------------
        X_train = self.embed_texts(train_df["job_description"].tolist())
        y_train = train_df["label"].values

        X_val = self.embed_texts(val_df["job_description"].tolist())
        y_val = val_df["label"].values

        # ------------------------------------------------------------
        # SAFE SMOTE (ONLY APPLY WHEN BOTH CLASSES EXIST)
        # ------------------------------------------------------------
        unique_classes = np.unique(y_train)
        logger.info(f"Classes found in training data: {unique_classes}")

        if len(unique_classes) > 1:
            from imblearn.over_sampling import SMOTE
            logger.info("Applying SMOTE oversampling...")
            sm = SMOTE(random_state=42)
            X_train_balanced, y_train_balanced = sm.fit_resample(X_train, y_train)
            logger.info(f"After SMOTE: {np.bincount(y_train_balanced)}")
        else:
            logger.warning("⚠ SMOTE skipped — only one class present in training data!")
            X_train_balanced, y_train_balanced = X_train, y_train

        # ------------------------------------------------------------
        # Train Logistic Regression
        # ------------------------------------------------------------
        logger.info("Training Logistic Regression model...")

        clf = LogisticRegression(max_iter=3000, class_weight="balanced")
        clf.fit(X_train_balanced, y_train_balanced)

        predictions = clf.predict(X_val)
        report = classification_report(y_val, predictions)
        logger.info("\n" + report)

        # ------------------------------------------------------------
        # Save models
        # ------------------------------------------------------------
        embedder_path = self.models_dir / "sentence_embedder"
        clf_path = self.models_dir / "job_fraud_classifier.pkl"

        self.embedder.save(str(embedder_path))
        pickle.dump(clf, open(clf_path, "wb"))

        logger.info("Embedding model saved.")
        logger.info(f"Classifier saved to: {clf_path}")

        return embedder_path, clf_path

# ============================================================
#              ARGS + MAIN FUNCTION
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2")
    return parser.parse_args()


def main():
    args = parse_args()

    config = {
        "model_name": args.model
    }

    trainer = ModelTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
