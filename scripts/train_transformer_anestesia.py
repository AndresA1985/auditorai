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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

from app.config import settings
from scripts.train_anestesia import DEFAULT_ALLOWED_TIMES, cargar_dataset, numeric_sort, texto_row
from scripts.train_model import dividir_indices, seleccionar

DEFAULT_MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'


class AnestesiaDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
        self.labels = np.asarray(labels, dtype=np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(value[idx]) for key, value in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def parse_values(value: str) -> set[str]:
    return {item.strip() for item in value.split(',') if item.strip()}


def preparar_particiones(texts, labels, validation_size, test_size):
    split_idx = dividir_indices(len(texts), validation_size, test_size)
    return {
        'train_texts': seleccionar(texts, split_idx['train']),
        'validation_texts': seleccionar(texts, split_idx['validation']),
        'test_texts': seleccionar(texts, split_idx['test']),
        'train_labels': seleccionar(labels, split_idx['train']),
        'validation_labels': seleccionar(labels, split_idx['validation']),
        'test_labels': seleccionar(labels, split_idx['test']),
        'split_idx': split_idx,
    }


def metricas(nombre: str, y_true, y_pred, classes: list[str]) -> dict:
    y_true_labels = [classes[int(idx)] for idx in y_true]
    y_pred_labels = [classes[int(idx)] for idx in y_pred]
    report = classification_report(y_true_labels, y_pred_labels, labels=classes, output_dict=True, zero_division=0)
    y_true_num = [float(value) for value in y_true_labels]
    y_pred_num = [float(value) for value in y_pred_labels]
    return {
        'name': nombre,
        'size': len(y_true_labels),
        'accuracy': round(float(accuracy_score(y_true_labels, y_pred_labels)), 4),
        'macro_f1': round(float(f1_score(y_true_labels, y_pred_labels, labels=classes, average='macro', zero_division=0)), 4),
        'weighted_f1': round(float(f1_score(y_true_labels, y_pred_labels, labels=classes, average='weighted', zero_division=0)), 4),
        'mae_hours': round(float(mean_absolute_error(y_true_num, y_pred_num)), 4),
        'labels': classes,
        'confusion_matrix': confusion_matrix(y_true_labels, y_pred_labels, labels=classes).tolist(),
        'classification_report': report,
    }


def crear_compute_metrics(classes: list[str]):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        pred = np.argmax(logits, axis=1)
        return {
            'accuracy': float(accuracy_score(labels, pred)),
            'macro_f1': float(f1_score(labels, pred, average='macro', zero_division=0)),
            'weighted_f1': float(f1_score(labels, pred, average='weighted', zero_division=0)),
        }
    return compute_metrics


def entrenar(args):
    allowed_times = parse_values(args.allowed_times)
    rows = cargar_dataset(args.limit, allowed_times)
    texts = [texto_row(row) for row in rows]
    labels = [row['tiempo_norm'] for row in rows]

    encoder = LabelEncoder()
    y_all = encoder.fit_transform(labels)
    classes = [str(value) for value in encoder.classes_]
    data = preparar_particiones(texts, list(y_all), args.validation_size, args.test_size)

    tokenizer = AutoTokenizer.from_pretrained(args.encoder_model)
    train_dataset = AnestesiaDataset(data['train_texts'], data['train_labels'], tokenizer, args.max_length)
    validation_dataset = AnestesiaDataset(data['validation_texts'], data['validation_labels'], tokenizer, args.max_length)
    test_dataset = AnestesiaDataset(data['test_texts'], data['test_labels'], tokenizer, args.max_length)

    id2label = {idx: label for idx, label in enumerate(classes)}
    label2id = {label: idx for idx, label in id2label.items()}
    model = AutoModelForSequenceClassification.from_pretrained(
        args.encoder_model,
        num_labels=len(classes),
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
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='eval_macro_f1',
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
        compute_metrics=crear_compute_metrics(classes),
    )
    trainer.train()

    validation_pred = np.argmax(trainer.predict(validation_dataset).predictions, axis=1)
    test_pred = np.argmax(trainer.predict(test_dataset).predictions, axis=1)

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))

    evaluation = {
        'evaluated': True,
        'target': 'tiempo_anestesia_por_codigo',
        'model': 'transformer_anestesia',
        'encoder_model': args.encoder_model,
        'loss': 'CrossEntropyLoss',
        'split': {
            'train_size': len(data['train_texts']),
            'validation_size': len(data['validation_texts']),
            'test_size': len(data['test_texts']),
            'train_fraction': round(len(data['train_texts']) / len(texts), 4),
            'validation_fraction': round(len(data['validation_texts']) / len(texts), 4),
            'test_fraction': round(len(data['test_texts']) / len(texts), 4),
        },
        'params': {
            'epochs': args.epochs,
            'learning_rate': args.learning_rate,
            'batch_size': args.batch_size,
            'eval_batch_size': args.eval_batch_size,
            'max_length': args.max_length,
        },
        'validation_metrics': metricas('validation', data['validation_labels'], validation_pred, classes),
        'final_test_metrics': metricas('test', data['test_labels'], test_pred, classes),
    }

    artifact = {
        'version': 1,
        'trained_at': datetime.now().isoformat(timespec='seconds'),
        'model': 'transformer_anestesia',
        'target': 'tiempo_anestesia_por_codigo',
        'encoder_model': args.encoder_model,
        'model_dir': str(model_dir.resolve()),
        'classes': classes,
        'max_length': args.max_length,
        'training_rows': len(rows),
        'unique_codes': len({row['codigo_norm'] for row in rows}),
        'allowed_times': numeric_sort(allowed_times),
        'evaluation': evaluation,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output)

    return {
        'output': str(output),
        'model_dir': str(model_dir.resolve()),
        'model': artifact['model'],
        'encoder_model': artifact['encoder_model'],
        'target': artifact['target'],
        'training_rows': artifact['training_rows'],
        'unique_codes': artifact['unique_codes'],
        'allowed_times': artifact['allowed_times'],
        'evaluation': evaluation,
    }


def main():
    parser = argparse.ArgumentParser(description='Fine-tuning Transformer multiclase para tiempo_anestesia por codigo.')
    parser.add_argument('--output', default='models/auditai_anestesia_transformer.joblib')
    parser.add_argument('--model-dir', default='models/auditai_anestesia_transformer_hf')
    parser.add_argument('--work-dir', default='models/auditai_anestesia_transformer_runs')
    parser.add_argument('--encoder-model', default=DEFAULT_MODEL)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--validation-size', type=float, default=settings.validation_size)
    parser.add_argument('--test-size', type=float, default=settings.test_size)
    parser.add_argument('--allowed-times', default=DEFAULT_ALLOWED_TIMES)
    parser.add_argument('--epochs', type=float, default=3.0)
    parser.add_argument('--learning-rate', type=float, default=2e-5)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--eval-batch-size', type=int, default=16)
    parser.add_argument('--max-length', type=int, default=384)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--logging-steps', type=int, default=25)
    parser.add_argument('--fp16', action='store_true')
    args = parser.parse_args()

    result = entrenar(args)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
