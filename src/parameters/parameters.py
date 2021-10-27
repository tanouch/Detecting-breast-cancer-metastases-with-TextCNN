from dataclasses import dataclass

@dataclass
class Parameters:
    #Preprocessing parameeters
    seq_len: int = 2048
    out_size: int = 5
    stride: int = 2
   
    #Training parameters
    epochs: int = 1201
    batch_size: int = 128
    learning_rate: float = 0.000003