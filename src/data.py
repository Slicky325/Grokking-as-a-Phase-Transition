import torch
from torch.utils.data import Dataset
import itertools
import random

class ModularArithmeticDataset(Dataset):
    """
    Dataset for Modular Arithmetic tasks: Addition, Division, and Multi-task.
    Modulo P is typically 53.
    """
    def __init__(self, P=53, task='addition', frac_train=0.3, seed=0, is_train=True):
        self.P = P
        self.task = task
        self.data = self._generate_data()
        
        random.seed(seed)
        indices = list(range(len(self.data)))
        random.shuffle(indices)
        
        split_idx = int(frac_train * len(self.data))
        if is_train:
            self.indices = indices[:split_idx]
        else:
            self.indices = indices[split_idx:]
            
        self.subset_data = [self.data[i] for i in self.indices]

    def _generate_data(self):
        # Token layout: 0..P-1 = numbers, P = op_add(+), P+1 = op_div(/), P+2 = eq(=)
        eq = self.P + 2
        op_add = self.P
        op_div = self.P + 1
        data = []
        if self.task == 'addition':
            for a, b in itertools.product(range(self.P), range(self.P)):
                tgt = (a + b) % self.P
                data.append((torch.tensor([a, b, eq], dtype=torch.long), torch.tensor(tgt, dtype=torch.long)))
        elif self.task == 'division':
            for a, b in itertools.product(range(self.P), range(1, self.P)):
                b_inv = pow(b, self.P - 2, self.P)
                tgt = (a * b_inv) % self.P
                data.append((torch.tensor([a, b, eq], dtype=torch.long), torch.tensor(tgt, dtype=torch.long)))
        elif self.task == 'multi-task':
            # Addition
            for a, b in itertools.product(range(self.P), range(self.P)):
                tgt = (a + b) % self.P
                data.append((torch.tensor([a, op_add, b, eq], dtype=torch.long), torch.tensor(tgt, dtype=torch.long)))
            # Division
            for a, b in itertools.product(range(self.P), range(1, self.P)):
                b_inv = pow(b, self.P - 2, self.P)
                tgt = (a * b_inv) % self.P
                data.append((torch.tensor([a, op_div, b, eq], dtype=torch.long), torch.tensor(tgt, dtype=torch.long)))
        else:
            raise ValueError(f"Unknown task: {self.task}")
            
        return data

    def __len__(self):
        return len(self.subset_data)

    def __getitem__(self, idx):
        return self.subset_data[idx]

def get_dataloaders(P=53, task='addition', batch_size=512, frac_train=0.3, seed=0):
    train_dataset = ModularArithmeticDataset(P, task, frac_train, seed, is_train=True)
    test_dataset = ModularArithmeticDataset(P, task, frac_train, seed, is_train=False)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader
