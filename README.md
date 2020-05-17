# LSTM Variants

### Installation
```bash
pip install lstm-variants
```
### Example usage
```python
from lstm_variants import LightweightLSTM
import torch

lstm = LightweightLSTM(input_size=5, hidden_size=32, batch_first=True)

batch_size, seq_len, input_size = 8, 35, 5

x = torch.rand(size=(batch_size, seq_len, input_size))

# Just like Pytorch's LSTM
output, (h_t, c_t) = lstm(x, states=None)
```