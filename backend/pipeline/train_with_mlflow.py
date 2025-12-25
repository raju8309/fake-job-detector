from pathlib import Path
import sys
import argparse
import json
import joblib
import pandas as pd

import mlflow

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score


def _add_backend_root_to_path(backend_root: Path) -> None:
    if str(backend_root) not in sys.path:
        sys.path.insert(0, str(backend_root))


def _safe_str(x) -> str:
    if x is None:
        return ""
    return str(x)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default=str(Path(__file__).resolve().parents[1] / "data" / "fake_job_postings.csv"),
    )
    parser.add_argument(
        "--experiment",
        default="fake-job-detector",
    )
    parser.add_argument(
        "--run-name",
        default=None,
    )
    parser.add_argument("--max-features", type=int, default=30000)
    parser.add_argument("--ngram-min", type=int, default=1)
    parser.add_argument("--ngram-max", type=int, default=2)
    parser.add_argument("--stop-words", type=str, default="english")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)

    parser.add_argument("--class-weight", type=str, default="balanced")
    parser.add_argument("--max-iter", type=int, default=1000)

    parser.add_argument(
        "--mlflow-uri",
        default="file:./mlruns",
        help="Use file:./mlruns for local tracking in backend/pipeline",
    )

    parser.add_argument(
        "--save-model",
        default=str(Path(__file__).resolve().parents[1] / "models" / "fake_job_model.pkl"),
    )
    parser.add_argument(
        "--save-vectorizer",
        default=str(Path(__file__).resolve().parents[1] / "models" / "tfidf_vectorizer.pkl"),
    )

    args = parser.parse_args()

    backend_root = Path(__file__).resolve().parents[1]
    _add_backend_root_to_path(backend_root)

    from app.utils.text_cleaning import clean_text

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {data_path}")

    df = pd.read_csv(data_path)
    if "fraudulent" not in df.columns:
        raise ValueError("Dataset missing required column: fraudulent")

    text = (
        df.get("title").apply(_safe_str)
        + " "
        + df.get("description").apply(_safe_str)
    ).str.strip()
    text_clean = text.apply(clean_text)

    y = df["fraudulent"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        text_clean,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    vectorizer = TfidfVectorizer(
        max_features=args.max_features,
        ngram_range=(args.ngram_min, args.ngram_max),
        stop_words=(None if args.stop_words.lower() in {"none", ""} else args.stop_words),
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(
        class_weight=(None if args.class_weight.lower() in {"none", ""} else args.class_weight),
        max_iter=args.max_iter,
        random_state=args.random_state,
        n_jobs=-1,
    )

    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.experiment)

    run_name = args.run_name
    if run_name is None:
        run_name = f"tfidf_lr_mf{args.max_features}_ng{args.ngram_min}-{args.ngram_max}_rs{args.random_state}"

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "max_features": args.max_features,
                "ngram_min": args.ngram_min,
                "ngram_max": args.ngram_max,
                "stop_words": args.stop_words,
                "test_size": args.test_size,
                "random_state": args.random_state,
                "class_weight": args.class_weight,
                "max_iter": args.max_iter,
                "rows": int(len(df)),
                "fraudulent_pos": int((y == 1).sum()),
                "fraudulent_neg": int((y == 0).sum()),
            }
        )

        model.fit(X_train_vec, y_train)

        y_pred = model.predict(X_test_vec)
        y_proba = None
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test_vec)[:, 1]

        acc = float(accuracy_score(y_test, y_pred))
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="binary", zero_division=0
        )

        metrics = {
            "accuracy": acc,
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
        }

        if y_proba is not None:
            try:
                metrics["roc_auc"] = float(roc_auc_score(y_test, y_proba))
            except Exception:
                pass

        mlflow.log_metrics(metrics)

        models_dir = Path(args.save_model).resolve().parent
        models_dir.mkdir(parents=True, exist_ok=True)

        model_path = Path(args.save_model).resolve()
        vectorizer_path = Path(args.save_vectorizer).resolve()

        joblib.dump(model, model_path)
        joblib.dump(vectorizer, vectorizer_path)

        mlflow.log_artifact(str(model_path), artifact_path="artifacts")
        mlflow.log_artifact(str(vectorizer_path), artifact_path="artifacts")

        mlflow.log_dict(metrics, "metrics.json")
        mlflow.log_dict(
            {
                "model_type": "LogisticRegression",
                "vectorizer_type": "TfidfVectorizer",
            },
            "run_info.json",
        )


if __name__ == "__main__":
    main()
