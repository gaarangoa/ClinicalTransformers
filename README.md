# Clinical Transformer
Clinical transformer framework for modeling

## Instalation
Download this repo and add it to your python environment: 

```python 
import sys
sys.path.append('/path/to/clinical_transformer/repository/')
```

## Usage 

### Training
```python
from xai.models import Trainer
from xai.models import SurvivalTransformer
from xai.models import OptimizedSurvivalDataGenerator as SurvivalDataGenerator
from xai.losses.survival import cIndex_SigmoidApprox as cindex_loss
from xai.metrics.survival import sigmoid_concordance as cindex

# setup parameters
set_seed(0)    
   
trainer = Trainer(
    out_dir = './out/',
    max_features_percentile=95,
    test_size=0.1,
    mode='survival',
    model=SurvivalTransformer, 
    dataloader=SurvivalDataGenerator,
    loss=cindex_loss,
    metrics=[cindex]
)

trainer.setup_data(
    data, 
    discrete_features = discrete_features,
    continuous_features = continuous_features,
    target=['time', 'event']
)

trainer.setup_model(
    learning_rate=0.0001,
    embedding_size=128,
    num_heads=4,
    num_layers=4,
    batch_size_max=True,
    save_best_only=False
)

trainer.fit(repetitions=10, epochs=100, verbose=0, seed=0)

```

### Value-masked self-supervision
```python
from xai.models import (
    Trainer,
    ValueMaskedSelfSupervisedTransformer,
    ValueMaskedSelfSupervisedDataGenerator,
)
from xai.losses.selfsupervision.value_mask import ValueMaskLoss

trainer = Trainer(
    out_dir="./out/value_mask/",
    max_features_percentile=90,
    test_size=0.0,
    mode="value-self-supervision",
    model=ValueMaskedSelfSupervisedTransformer,
    dataloader=ValueMaskedSelfSupervisedDataGenerator,
    loss=ValueMaskLoss(),
)

trainer.setup_data(
    data,
    discrete_features=discrete_features,
    continuous_features=continuous_features,
    target=[],
)

trainer.setup_model(
    learning_rate=5e-4,
    embedding_size=128,
    num_heads=4,
    num_layers=4,
    batch_size_max=True,
)

trainer.fit(repetitions=1, epochs=200, verbose=1)
```

### PyTorch / ctv2 Value-Masked SSL
```python
import pandas as pd
import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator

from ctv2 import (
    TabularTokenizer,
    FeatureTokenizer,
    ValueMaskedAnnDataDataset,
    value_mask_collate,
    ValueMaskedConfig,
    ValueMaskedTransformer,
    ValueMaskedTrainer,
)

raw = pd.read_csv("cohort.csv")
tab_tokenizer = TabularTokenizer(
    categorical_features=["sex", "stage"],
    binary_features=["smoker"],
    continuous_features=["age", "ldh", "bmi"],
).fit(raw)
adata = tab_tokenizer.transform(raw)

tokenizer = FeatureTokenizer(adata.var_names)
dataset = ValueMaskedAnnDataDataset(adata, tokenizer, max_features=256, mask_fraction=0.2)
loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=value_mask_collate)

config = ValueMaskedConfig(vocab_size=tokenizer.vocab_size, max_seq_len=dataset.seq_len)
model = ValueMaskedTransformer(config, pad_token_id=tokenizer.pad_id)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

trainer = ValueMaskedTrainer(model, accelerator=Accelerator())
trainer.fit(loader, optimizer, epochs=10)
```

### Predicting 
```python
from xai.models.explainer import SurvivalExtractor
from lifelines.utils import concordance_index as lfcindex

model = SurvivalExtractor(
    fold=fold, 
    time=time, 
    event=event, 
    sample_id=sample_id,
    epoch=epoch, 
    path=path,
)

set_seed(0)
data = model.scores(data, iterations=2)
train_ci = lfcindex(data[time], data['β'], data[event])

```
