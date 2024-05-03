import torch
import torch.nn as nn

batch_size = 4
seq_len = 64
input_size = 192

# Assuming you have your input sequence `seq_data`
seq_data = torch.rand((batch_size, seq_len, input_size))  # (batch_size, seq_len, input_size)

lstm = nn.LSTM(input_size, input_size, batch_first=True)

# Initialize hidden and cell states
h0 = torch.zeros(1, batch_size, input_size)
c0 = torch.zeros(1, batch_size, input_size)

# Outputs for sequential processing
seq_outputs = []

h = h0
c = c0
for step in range(seq_data.size(1)):
    # Feed current input and previous hidden/cell states
    output, (h, c) = lstm(seq_data[:, step, ...].unsqueeze(1), (h.detach(), c.detach()))
    seq_outputs.append(output)

seq_outputs = torch.cat(seq_outputs, dim=1)
outputs_all_steps, (h_n, c_n) = lstm(seq_data, (h0, c0))

assert torch.all(torch.isclose(seq_outputs, outputs_all_steps))
# seq_outputs should now be the same as if you processed the entire sequence at once
