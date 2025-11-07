# Clinical Transformer
Clinical transformer framework for modeling 

![xAI Framework](xAI.png)

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
