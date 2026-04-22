import torch
import torch.nn as nn
from torch.optim import AdamW
import pandas as pd

from .metrics import compute_llc, compute_order_parameter

def should_log(epoch: int) -> bool:
    """ Logarithmic checking transitioning to linear. """
    if epoch <= 1000:
        return epoch in [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    else:
        return epoch % 100 == 0

def train_model(model, train_loader, test_loader, epochs=20000, lr=1e-3,
                weight_decay=1.0, P=53, skip_llc=False) -> pd.DataFrame:
    """
    Custom training loop using AdamW with logarithmic checkpointing.
    At checkpoints, optimizer is paused, evaluates model, computes losses, and gets metrics.
    """
    device = next(model.parameters()).device
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.98))
    criterion = nn.CrossEntropyLoss()
    
    results = []
    
    model.train()
    for epoch in range(1, epochs + 1):
        # Training step
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            
        # Logging schedule at checkpoints
        if should_log(epoch):
            model.eval() # Pause optimizer implicitly, set to eval
            
            with torch.no_grad():
                # Compute train loss over the whole dataset
                train_loss = 0.0
                train_total = 0
                for x, y in train_loader:
                    x, y = x.to(device), y.to(device)
                    out = model(x)
                    train_loss += criterion(out, y).item() * x.size(0)
                    train_total += x.size(0)
                train_loss /= train_total
                
                # Compute test loss
                test_loss = 0.0
                test_total = 0
                for x, y in test_loader:
                    x, y = x.to(device), y.to(device)
                    out = model(x)
                    test_loss += criterion(out, y).item() * x.size(0)
                    test_total += x.size(0)
                test_loss /= test_total
                
            # Physics Metrics
            if skip_llc:
                llc = float('nan')
            else:
                try:
                    llc = compute_llc(model, train_loader, criterion)
                except Exception as e:
                    print(f"LLC computation failed: {e}")
                    llc = float('nan')
            
            try:
                order_param = compute_order_parameter(model, P=P)
            except Exception as e:
                print(f"Order parameter computation failed: {e}")
                order_param = float('nan')
                
            print(f"Epoch {epoch:5d} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | LLC: {llc:.4f} | Order Param: {order_param:.4f}")
            
            results.append({
                'Epoch': epoch,
                'Train_Loss': train_loss,
                'Test_Loss': test_loss,
                'LLC': llc,
                'Order_Parameter': order_param
            })
            model.train() # Resume training

    df = pd.DataFrame(results)
    return df
