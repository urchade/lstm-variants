# LSTM Variants

### Installation
```bash
pip install lstm-variants
```
### Usage
```python
from lstm_variants import LightweightLSTM
import torch

module = LightweightLSTM(input_size=5, hidden_size=32, batch_first=True)

batch_size, seq_len, input_size = 8, 35, 5

x = torch.rand(size=(batch_size, seq_len, input_size))

output, (h_t, c_t) = module(x)
```