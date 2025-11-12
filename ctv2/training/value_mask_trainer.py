from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from accelerate import Accelerator
import torch
from torch.utils.data import DataLoader

from ctv2.losses import value_mask_loss
from ctv2.models import ValueMaskedTransformer


@dataclass
class TrainingState:
    epoch: int
    global_step: int
    loss: float


class ValueMaskedTrainer:
    """
    Minimal training harness that mirrors the behavior of the TensorFlow Trainer
    but runs entirely on PyTorch + Accelerate.
    """

    def __init__(self, model: ValueMaskedTransformer, accelerator: Optional[Accelerator] = None) -> None:
        self.model = model
        self.accelerator = accelerator or Accelerator()

    def fit(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        *,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 1,
        gradient_accumulation_steps: int = 1,
    ):
        accelerator = self.accelerator
        if val_loader is not None:
            model, optimizer, train_loader, val_loader = accelerator.prepare(
                self.model, optimizer, train_loader, val_loader
            )
        else:
            model, optimizer, train_loader = accelerator.prepare(
                self.model, optimizer, train_loader
            )

        global_step = 0
        history = []

        for epoch in range(epochs):
            model.train()
            for step, batch in enumerate(train_loader):
                preds = model(
                    token_ids=batch["token_ids"],
                    values=batch["values"],
                    value_mask=batch["value_mask"],
                    padding_mask=batch["padding_mask"],
                )
                loss = value_mask_loss(
                    preds, batch["targets"], batch["value_mask"]
                ) / gradient_accumulation_steps

                accelerator.backward(loss)
                if (step + 1) % gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                global_step += 1

            eval_loss = None
            if val_loader is not None:
                model.eval()
                losses = []
                with torch.no_grad():
                    for batch in val_loader:
                        preds = model(
                            token_ids=batch["token_ids"],
                            values=batch["values"],
                            value_mask=batch["value_mask"],
                            padding_mask=batch["padding_mask"],
                        )
                        loss = value_mask_loss(
                            preds, batch["targets"], batch["value_mask"]
                        )
                        losses.append(accelerator.gather(loss.detach()))
                if losses:
                    eval_loss = torch.cat(losses).mean().item()

            state = TrainingState(
                epoch=epoch,
                global_step=global_step,
                loss=loss.detach().item() * gradient_accumulation_steps,
            )
            if eval_loss is not None:
                state.eval_loss = eval_loss  # type: ignore[attr-defined]
            history.append(state)

        return history
