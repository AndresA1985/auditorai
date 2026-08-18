import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import torch
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

from app.config import settings
from scripts.train_model import dividir_indices, seleccionar
from scripts.train_multilabel import (
    DEFAULT_THRESHOLDS,
    VALID_SELECTION_METRICS,
    evaluar_probabilidades,
    metricas_multilabel,
    parse_float_list,
    preparar_dataset,
    seleccionar_mejor,
)

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class AuditoriaDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
        self.labels = labels.astype(np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(value[idx]) for key, value in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float32)
        return item


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))

def preparar_particiones(texts, labels, validation_size, test_size):
    split_idx = dividir_indices(len(texts), validation_size, test_size)
    return {
        "train_texts": seleccionar(texts, split_idx["train"]),
        "validation_texts": seleccionar(texts, split_idx["validation"]),
        "test_texts": seleccionar(texts, split_idx["test"]),
        "train_labels": seleccionar(labels, split_idx["train"]),
        "validation_labels": seleccionar(labels, split_idx["validation"]),
        "test_labels": seleccionar(labels, split_idx["test"]),
        "split_idx": split_idx,
    }


def crear_compute_metrics(threshold: float):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probabilities = sigmoid(logits)
        pred = (probabilities >= threshold).astype(int)
        return metricas_multilabel(labels.astype(int), pred)

    return compute_metrics


def aplicar_min_labels(probabilities, pred, min_labels):
    if min_labels <= 0:
        return pred
    for row_idx, row in enumerate(probabilities):
        if pred[row_idx].sum() < min_labels:
            top_idx = np.argpartition(row, -min_labels)[-min_labels:]
            pred[row_idx, top_idx] = 1
    return pred

def entrenar(args):
    texts, labels = preparar_dataset(args.limit)
    data = preparar_particiones(texts, labels, args.validation_size, args.test_size)

    mlb = MultiLabelBinarizer()
    y_all = mlb.fit_transform(labels)
    y_train = y_all[data["split_idx"]["train"]]
    y_validation = y_all[data["split_idx"]["validation"]]
    y_test = y_all[data["split_idx"]["test"]]

    tokenizer = AutoTokenizer.from_pretrained(args.encoder_model)
    train_dataset = AuditoriaDataset(data["train_texts"], y_train, tokenizer, args.max_length)
    validation_dataset = AuditoriaDataset(data["validation_texts"], y_validation, tokenizer, args.max_length)
    test_dataset = AuditoriaDataset(data["test_texts"], y_test, tokenizer, args.max_length)

    id2label = {idx: str(label) for idx, label in enumerate(mlb.classes_)}
    label2id = {label: idx for idx, label in id2label.items()}
    model = AutoModelForSequenceClassification.from_pretrained(
        args.encoder_model,
        num_labels=len(mlb.classes_),
        problem_type="multi_label_classification",
        id2label=id2label,
        label2id=label2id,
    )

    training_args = TrainingArguments(
        output_dir=str(args.work_dir),
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        weight_decay=args.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_" + args.selection_metric,
        greater_is_better=True,
        logging_steps=args.logging_steps,
        report_to=[],
        fp16=args.fp16,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        tokenizer=tokenizer,
        compute_metrics=crear_compute_metrics(args.metric_threshold),
    )
    trainer.train()

    validation_logits = trainer.predict(validation_dataset).predictions
    validation_probabilities = sigmoid(validation_logits)
    validation_grid = evaluar_probabilidades(validation_probabilities, y_validation, args.thresholds, args.min_labels)
    validation_best = seleccionar_mejor(validation_grid, args.selection_metric)

    test_logits = trainer.predict(test_dataset).predictions
    test_probabilities = sigmoid(test_logits)
    test_pred = (test_probabilities >= validation_best["threshold"]).astype(int)
    test_pred = aplicar_min_labels(test_probabilities, test_pred, args.min_labels)

    final_test_metrics = {
        "name": "test",
        "size": len(data["test_texts"]),
        "threshold": validation_best["threshold"],
        "min_labels": args.min_labels,
        **metricas_multilabel(y_test, test_pred),
    }

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))

    evaluation = {
        "evaluated": True,
        "target": "codigo_grupo_auditor",
        "model": "transformer_multilabel",
        "encoder_model": args.encoder_model,
        "loss": "BCEWithLogitsLoss",
        "selection_metric": args.selection_metric,
        "split": {
            "train_size": len(data["train_texts"]),
            "validation_size": len(data["validation_texts"]),
            "test_size": len(data["test_texts"]),
            "train_fraction": round(len(data["train_texts"]) / len(texts), 4),
            "validation_fraction": round(len(data["validation_texts"]) / len(texts), 4),
            "test_fraction": round(len(data["test_texts"]) / len(texts), 4),
        },
        "best_params": {
            "threshold": validation_best["threshold"],
            "min_labels": args.min_labels,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
        },
        "validation_metrics": validation_best,
        "validation_grid": validation_grid,
        "final_test_metrics": final_test_metrics,
    }

    artifact = {
        "version": 1,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "model": "transformer_multilabel",
        "target": "codigo_grupo_auditor",
        "encoder_model": args.encoder_model,
        "model_dir": str(model_dir.resolve()),
        "classes": [str(label) for label in mlb.classes_],
        "threshold": validation_best["threshold"],
        "min_labels": args.min_labels,
        "max_length": args.max_length,
        "training_rows": len(texts),
        "unique_codes": len(mlb.classes_),
        "evaluation": evaluation,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output)

    return {
        "output": str(output),
        "model_dir": str(model_dir.resolve()),
        "model": artifact["model"],
        "encoder_model": artifact["encoder_model"],
        "target": artifact["target"],
        "training_rows": artifact["training_rows"],
        "unique_codes": artifact["unique_codes"],
        "evaluation": evaluation,
    }


def main():
    parser = argparse.ArgumentParser(description="Fine-tuning Transformer multi-label para codigo_grupo_auditor.")
    parser.add_argument("--output", default="models/auditai_transformer_multilabel.joblib")
    parser.add_argument("--model-dir", default="models/auditai_transformer_multilabel_hf")
    parser.add_argument("--work-dir", default="models/auditai_transformer_runs")
    parser.add_argument("--encoder-model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--validation-size", type=float, default=settings.validation_size)
    parser.add_argument("--test-size", type=float, default=settings.test_size)
    parser.add_argument("--thresholds", type=parse_float_list, default=parse_float_list(DEFAULT_THRESHOLDS))
    parser.add_argument("--selection-metric", choices=sorted(VALID_SELECTION_METRICS), default="avg_dice")
    parser.add_argument("--metric-threshold", type=float, default=0.5)
    parser.add_argument("--min-labels", type=int, default=1)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()

    result = entrenar(args)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
