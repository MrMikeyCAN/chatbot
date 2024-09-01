import dataset
import model
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import optuna
import audio
import time

input_size = audio.N_MELS
output_size = len(dataset.train_dataset.alphabet) + 1

if dataset.device == torch.device('cuda'):
    torch.cuda.empty_cache()


def train(trial):
    hidden_size = trial.suggest_int('hidden_size', 128, 256)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
    beta1 = trial.suggest_float('beta1', 0.8, 0.99, log=True)
    beta2 = trial.suggest_float('beta2', 0.9, 0.9999, log=True)
    factor = trial.suggest_float('factor', 0.1, 0.5)
    patience = trial.suggest_int('patience', 10, 100)
    epochs = trial.suggest_int('epochs', 10, 1000)

    _model = model.Model(input_size, hidden_size, output_size, dropout).to(dataset.device)

    total_params = sum(p.numel() for p in _model.parameters())
    print('Total number of parameters: {}'.format(total_params))

    criterion = nn.CTCLoss()
    optimizer = optim.Adam(_model.parameters(), lr=learning_rate, betas=[beta1, beta2], weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience)

    best_val_loss = float('inf')
    times = []
    for epoch in range(epochs):

        _model.train()
        train_loss = 0
        for i, (x, y, input_lengths, target_lengths) in enumerate(dataset.train_dataloader, start=1):
            optimizer.zero_grad()
            y_pred = _model(x).transpose(0, 1).log_softmax(2)

            loss = criterion(y_pred, y, input_lengths, target_lengths)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            if dataset.device == torch.device('cuda'):
                torch.cuda.empty_cache()

        train_loss /= i

        _model.eval()
        val_loss = 0
        with torch.no_grad():
            for i, (x, y, input_lengths, target_lengths) in enumerate(dataset.dev_dataloader, start=1):
                start_time = time.time()
                y_pred = _model(x).transpose(0, 1).log_softmax(2)
                end_time = time.time()
                loss = criterion(y_pred, y, input_lengths, target_lengths)
                val_loss += loss.item()
                times.append(end_time - start_time)

                if dataset.device == torch.device('cuda'):
                    torch.cuda.empty_cache()

        val_loss /= i

        scheduler.step(val_loss)

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, Learning Rate: {scheduler.get_last_lr()}, "
              f"Time: {np.mean(times):.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(_model.state_dict(), 'best_model.pth')

        if dataset.device == torch.device('cuda'):
            torch.cuda.empty_cache()

    avg_time = np.mean(times)
    print(f"Average time: {avg_time:.2f}s")

    return best_val_loss, avg_time


def objective(trial):
    val_loss, avg_time = train(trial)
    trial.set_user_attr('avg_time', avg_time)
    return val_loss if avg_time <= MAX_AVG_TIME else float('inf')


MAX_AVG_TIME = 0.1  # 100 ms
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=25)

print("Best trial:")
trial = study.best_trial
print("  Value: ", trial.value)
print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")
print(f"  Average time: {trial.user_attrs['avg_time']:.2f}s")

# trial.params['dropout'] eklenmiyor test setinde
best_model = model.Model(input_size, trial.params['hidden_size'], trial.params['num_layers'], output_size, 0).to(
    dataset.device)
best_model.load_state_dict(torch.load('best_model.pth'))

best_model.eval()
test_loss = 0
criterion = nn.CTCLoss()
times = []
with torch.no_grad():
    for x, y, input_lengths, target_lengths in dataset.test_dataloader:
        start_time = time.time()
        y_pred = best_model(x).transpose(0, 1).log_softmax(2)
        end_time = time.time()
        loss = criterion(y_pred, y, input_lengths, target_lengths)
        test_loss += loss.item()
        times.append(end_time - start_time)

test_loss /= len(dataset.test_dataloader)
print(f"Test Loss: {test_loss:.4f}, Time: {np.mean(times):.4f}")
