"""
Linear probing sobre FinBERT: backbone congelado + cabeza clasificadora
entrenada sobre labels manuales.

Justificación metodológica: con ~2000-3000 posts etiquetados a mano, un
fine-tune completo del backbone (110M parámetros) sobreajusta. Se congela
el backbone de ProsusAI/finbert y solo se entrena una cabeza lineal sobre
el embedding [CLS], que tiene muchas órdenes de magnitud menos parámetros.

Solo se persiste la cabeza (models/finbert_finetuned/); el backbone se
recarga desde HuggingFace cada vez (finbert_finetuned_scorer.py hace lo
mismo para inferencia).
"""

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from transformers import AutoModel, AutoTokenizer

from config.settings import (
    FINBERT_FINETUNED_MODEL_PATH,
    FINBERT_MAX_LENGTH,
    FINBERT_MODEL_NAME,
    FINETUNE_BATCH_SIZE,
    FINETUNE_EPOCHS,
    FINETUNE_LR_HEAD,
    LABELED_STORE_FILE,
    SEED,
)
from src.labeling.labeled_store import load as load_labels

logger = logging.getLogger(__name__)

# Mapeo de clase a índice para las tres categorías del golden set manual.
# Mismo orden que _IDX_TO_LABEL de finbert_scorer.py para que las métricas
# entre enfoques sean directamente comparables.
_LABEL_TO_IDX: dict[str, int] = {"positive": 0, "negative": 1, "neutral": 2}
_IDX_TO_LABEL: dict[int, str] = {v: k for k, v in _LABEL_TO_IDX.items()}


def _iter_batches(items: list, size: int):
    """Genera sub-listas de longitud máxima size."""
    for start in range(0, len(items), size):
        yield items[start: start + size]


# ---------------------------------------------------------------------------
# Hiperparámetros
# ---------------------------------------------------------------------------

@dataclass
class FineTuneHyperparams:
    """Hiperparámetros del entrenamiento de la cabeza clasificadora."""

    lr_head: float = FINETUNE_LR_HEAD
    epochs: int = FINETUNE_EPOCHS
    batch_size: int = FINETUNE_BATCH_SIZE
    weight_decay: float = 0.01
    max_length: int = FINBERT_MAX_LENGTH
    val_split: float = 0.15
    seed: int = SEED


# ---------------------------------------------------------------------------
# Cabeza clasificadora
# ---------------------------------------------------------------------------

class FinBERTClassifierHead(nn.Module):
    """Cabeza lineal sobre el embedding [CLS] del backbone congelado."""

    def __init__(self, hidden_size: int = 768, num_labels: int = 3, dropout: float = 0.1) -> None:
        """
        Args:
            hidden_size: Dimensión del embedding [CLS] (768 en FinBERT base).
            num_labels: Cantidad de clases de salida (3: positive/negative/neutral).
            dropout: Probabilidad de dropout aplicada antes de la capa lineal.
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, num_labels)

    def forward(self, cls_embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cls_embedding: Tensor [batch, hidden_size] del token [CLS].

        Returns:
            Logits sin normalizar [batch, num_labels].
        """
        return self.linear(self.dropout(cls_embedding))


# ---------------------------------------------------------------------------
# Entrenador
# ---------------------------------------------------------------------------

class FinBERTHeadTrainer:
    """
    Encapsula el backbone congelado, la cabeza entrenable y el loop de
    entrenamiento por linear probing.
    """

    def __init__(
        self,
        hparams: FineTuneHyperparams = FineTuneHyperparams(),
        model_name: str = FINBERT_MODEL_NAME,
    ) -> None:
        """
        Carga el backbone congelado y crea la cabeza clasificadora.

        Args:
            hparams: Hiperparámetros de entrenamiento.
            model_name: Identificador HuggingFace del backbone (ProsusAI/finbert).
        """
        self._hparams = hparams
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info("Cargando backbone FinBERT ('%s') congelado en %s...", model_name, self._device)
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._backbone = AutoModel.from_pretrained(model_name)
        self._backbone.to(self._device)
        self._backbone.eval()
        for param in self._backbone.parameters():
            param.requires_grad_(False)

        hidden_size = self._backbone.config.hidden_size
        self.head = FinBERTClassifierHead(hidden_size=hidden_size, num_labels=len(_LABEL_TO_IDX))
        self.head.to(self._device)
        logger.info("Cabeza clasificadora inicializada (hidden_size=%d).", hidden_size)

    # ------------------------------------------------------------------
    # Internos
    # ------------------------------------------------------------------

    def _embed_batch(self, texts: list[str]) -> torch.Tensor:
        """Tokeniza y pasa un batch por el backbone congelado; devuelve el embedding [CLS]."""
        encoding = self._tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self._hparams.max_length,
        )
        encoding = {k: v.to(self._device) for k, v in encoding.items()}
        with torch.no_grad():
            output = self._backbone(**encoding)
        return output.last_hidden_state[:, 0, :]

    def _compute_class_weights(self, labels: list[int]) -> torch.Tensor:
        """Pesos de clase inversamente proporcionales a la frecuencia en `labels`."""
        counts = np.bincount(labels, minlength=len(_LABEL_TO_IDX)).astype(float)
        counts = np.where(counts == 0, 1.0, counts)  # evita div/0 si falta una clase
        weights = counts.sum() / (len(counts) * counts)
        return torch.tensor(weights, dtype=torch.float32, device=self._device)

    def _evaluate(
        self,
        texts: list[str],
        labels: list[int],
        criterion: nn.Module,
    ) -> tuple[float, float]:
        """Evalúa loss y accuracy de la cabeza en modo eval, sin actualizar pesos."""
        self.head.eval()
        total_loss, n_batches, correct = 0.0, 0, 0
        with torch.no_grad():
            for batch_texts, batch_labels in zip(
                _iter_batches(texts, self._hparams.batch_size),
                _iter_batches(labels, self._hparams.batch_size),
            ):
                embeddings = self._embed_batch(batch_texts)
                targets = torch.tensor(batch_labels, dtype=torch.long, device=self._device)
                logits = self.head(embeddings)
                loss = criterion(logits, targets)
                total_loss += loss.item()
                n_batches += 1
                correct += int((logits.argmax(dim=-1) == targets).sum().item())
        avg_loss = total_loss / max(n_batches, 1)
        accuracy = correct / max(len(texts), 1)
        return avg_loss, accuracy

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def train(self, texts: list[str], labels: list[str]) -> dict:
        """
        Entrena la cabeza clasificadora sobre labels manuales.

        Divide train/val de forma estratificada, entrena con AdamW y
        class weighting inversamente proporcional a la frecuencia de cada
        clase (compensa el desbalance típico de datasets financieros, donde
        'neutral' suele dominar).

        Args:
            texts: Textos limpios (misma limpieza que preprocessor.clean_text).
            labels: Labels manuales, una por texto ('positive'|'negative'|'neutral').

        Returns:
            Historial de entrenamiento: {"train_loss": [...], "val_loss": [...],
            "val_accuracy": [...]}, una entrada por época.

        Raises:
            ValueError: si texts y labels no tienen la misma longitud o hay
                labels fuera de _LABEL_TO_IDX.
        """
        if len(texts) != len(labels):
            raise ValueError(
                f"texts y labels deben tener la misma longitud "
                f"({len(texts)} != {len(labels)})"
            )
        unknown = set(labels) - set(_LABEL_TO_IDX)
        if unknown:
            raise ValueError(
                f"Labels desconocidas: {unknown}. Esperadas: {set(_LABEL_TO_IDX)}"
            )

        label_idx = [_LABEL_TO_IDX[l] for l in labels]

        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts,
            label_idx,
            test_size=self._hparams.val_split,
            random_state=self._hparams.seed,
            stratify=label_idx,
        )
        logger.info(
            "Split: %d train / %d val (val_split=%.2f).",
            len(train_texts), len(val_texts), self._hparams.val_split,
        )

        class_weights = self._compute_class_weights(train_labels)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = AdamW(
            self.head.parameters(),
            lr=self._hparams.lr_head,
            weight_decay=self._hparams.weight_decay,
        )

        history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "val_accuracy": []}

        for epoch in range(self._hparams.epochs):
            self.head.train()
            epoch_loss, n_batches = 0.0, 0
            for batch_texts, batch_labels in zip(
                _iter_batches(train_texts, self._hparams.batch_size),
                _iter_batches(train_labels, self._hparams.batch_size),
            ):
                embeddings = self._embed_batch(batch_texts)
                targets = torch.tensor(batch_labels, dtype=torch.long, device=self._device)

                optimizer.zero_grad()
                logits = self.head(embeddings)
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            train_loss = epoch_loss / max(n_batches, 1)
            val_loss, val_accuracy = self._evaluate(val_texts, val_labels, criterion)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_accuracy"].append(val_accuracy)

            logger.info(
                "Epoch %d/%d: train_loss=%.4f val_loss=%.4f val_accuracy=%.4f",
                epoch + 1, self._hparams.epochs, train_loss, val_loss, val_accuracy,
            )

        return history

    def save(self, output_dir: Path = FINBERT_FINETUNED_MODEL_PATH) -> Path:
        """
        Persiste solo la cabeza entrenada (no el backbone).

        Args:
            output_dir: Directorio donde se guardan head.pt y meta.json.

        Returns:
            Ruta al directorio de salida.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.head.state_dict(), output_dir / "head.pt")

        meta = {
            "hidden_size": self._backbone.config.hidden_size,
            "label_to_idx": _LABEL_TO_IDX,
            "hparams": asdict(self._hparams),
        }
        (output_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        logger.info("Cabeza persistida en %s (el backbone no se guarda).", output_dir)
        return output_dir

    @classmethod
    def load_head(
        cls,
        model_dir: Path = FINBERT_FINETUNED_MODEL_PATH,
        model_name: str = FINBERT_MODEL_NAME,
    ) -> "FinBERTHeadTrainer":
        """
        Reconstruye el backbone desde HuggingFace y le monta la cabeza persistida.

        Args:
            model_dir: Directorio con head.pt y meta.json (salida de save()).
            model_name: Identificador HuggingFace del backbone a recargar.

        Returns:
            Instancia de FinBERTHeadTrainer lista para inferencia (head en eval()).

        Raises:
            FileNotFoundError: si no existe meta.json en model_dir.
        """
        meta_path = model_dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"No se encontró {meta_path}. Corré train() + save() primero."
            )
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        hparams = FineTuneHyperparams(**meta["hparams"])

        instance = cls(hparams=hparams, model_name=model_name)
        state_dict = torch.load(model_dir / "head.pt", map_location=instance._device)
        instance.head.load_state_dict(state_dict)
        instance.head.eval()

        logger.info("Cabeza cargada desde %s.", model_dir)
        return instance


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    store_path = Path(sys.argv[1]) if len(sys.argv) > 1 else LABELED_STORE_FILE
    labels_df = load_labels(store_path)
    if labels_df.empty:
        print(
            f"No hay labels manuales en {store_path}. Etiquetá posts primero "
            "(ver src/labeling/uncertainty_sampling.py)."
        )
        sys.exit(1)

    trainer = FinBERTHeadTrainer()
    history = trainer.train(labels_df["text"].tolist(), labels_df["label"].tolist())
    result_dir = trainer.save()

    print(f"Cabeza entrenada y guardada en → {result_dir}")
    print(
        f"Última época: train_loss={history['train_loss'][-1]:.4f} "
        f"val_loss={history['val_loss'][-1]:.4f} "
        f"val_accuracy={history['val_accuracy'][-1]:.4f}"
    )
