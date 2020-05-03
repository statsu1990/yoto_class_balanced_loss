import pandas as pd
import torch

def save_log(logs, columns, filename):
    df = pd.DataFrame(logs)
    df.columns = columns
    df.to_csv(filename)
    return

def save_checkpoint(epoch, model, optimizer, file_name):
    state = {'epoch': epoch + 1, 'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict(), }
    torch.save(state, file_name)
    return

