from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score, accuracy_score, classification_report
from torch.utils.data import DataLoader
from tqdm import tqdm


def train(
    model,
    loss_func,
    train_dataloader: DataLoader,
    valid_dataloader: DataLoader,
    epochs=10,
    checkpoint_file=None,
):

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    device = next(model.parameters()).device
    print(f"{device} device for training")
    # initialize running values
    history = {
        "train_loss": [],
        "valid_loss": [],
        "acc": [],
        "f1": [],
        "epoch": [],
        "step": [],
    }
    best_acc = 0
    best_acc_epoch = 0
    nr_non_improving = 0
    patience = 3

    def evaluate_step(step, epoch, running_loss, max_eval_steps=None):
        nonlocal best_acc, best_acc_epoch, nr_non_improving
        if valid_dataloader is None:
            return

        print(f"Evaluating (Epoch/Step) {epoch +1} / {step}: ")
        average_train_loss = running_loss / step
        valid_loss, valid_acc, valid_f1 = evaluate(
            model, valid_dataloader, loss_func, epoch, max_eval_steps
        )
        history["epoch"].append(epoch)
        history["step"].append(step)
        history["train_loss"].append(average_train_loss)
        history["valid_loss"].append(valid_loss)
        history["acc"].append(valid_acc)
        history["f1"].append(valid_f1)

        print(
            f"Epoch [{epoch + 1}/{epochs}] - step {step}, Train Loss: {average_train_loss}, "
            f"Val Loss: {valid_loss}, Val Acc:{valid_acc}, Val F1:{valid_f1}"
        )

        if round(best_acc, 5) < round(valid_acc, 5):
            best_acc = valid_acc
            best_acc_epoch = epoch
            if checkpoint_file is not None:
                model.save(
                    checkpoint_file,
                    train_dataloader.dataset.diacritics_vocab,
                    epoch,
                    valid_acc,
                    valid_f1,
                )
            nr_non_improving = 0
        else:
            nr_non_improving += 1
            print(f"Accuracy did not improve {best_acc} < {valid_acc}")

    if Path(checkpoint_file).exists():
        # Evaluate the existing model
        evaluate_step(1, 0, 0, 300_000 // valid_dataloader.batch_size)

    for epoch in range(epochs):
        model.train()

        step = 0
        running_loss = 0.0
        optimizer.zero_grad()

        epoch_predictions = []
        epoch_true_labels = []

        for (char_input, word_emb, sentence_emb), labels in train_dataloader:
            labels = labels.to(device)
            logits = model(
                char_input.to(device), word_emb.to(device), sentence_emb.to(device)
            )

            loss = loss_func(logits, labels)

            _, predicted = torch.max(logits, -1)
            epoch_predictions.extend(predicted.tolist())
            epoch_true_labels.extend([np.argmax(x) for x in labels.tolist()])

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # update running values
            running_loss += loss.item()
            step += 1

            if step % 1000 == 0:
                epoch_accuracy = accuracy_score(epoch_true_labels, epoch_predictions)
                epoch_f1_metrics = f1_score(
                    np.array(epoch_true_labels).reshape(-1),
                    np.array(epoch_predictions).reshape(-1),
                    average="weighted",
                )
                average_train_loss = running_loss / step
                print(
                    f"Epoch {epoch+1} / Step: {step},
                    Loss: {loss.item()},
                    Avg Loss: {average_train_loss} ,
                    Train ACC: {epoch_accuracy},
                    Train F1: {epoch_f1_metrics}",
                )

            if step % 10000 == 0:
                evaluate_step(
                    step, epoch, running_loss, 300_000 // valid_dataloader.batch_size
                )
                model.train()
                if nr_non_improving >= patience:
                    print(f"Early stopping, patience ({patience}) ran out ")
                    break
        # print progress
        print(f"--------------------------- Epoch {epoch+1} -------------------------")
        evaluate_step(step, epoch, running_loss)

    if valid_dataloader is None and checkpoint_file is not None:
        model.save(checkpoint_file, train_dataloader.dataset.diacritics_vocab, epoch)

    print(f"Finished Training! Best Epoch = {best_acc_epoch}")


def evaluate(model, dataloader: DataLoader, loss_func, epoch=None, max_eval_steps=None):
    # print("***** Running prediction *****")
    model.eval()
    predict_out = []
    all_label_ids = []
    eval_loss = 0
    total = 0
    correct = 0
    steps = 0
    device = next(model.parameters()).device
    with torch.no_grad():
        for (char_input, word_emb, sentence_emb), labels in tqdm(
            dataloader, total=max_eval_steps
        ):
            labels = labels.to(device)
            logits = model(
                char_input.to(device), word_emb.to(device), sentence_emb.to(device)
            )

            loss = loss_func(logits, labels)

            eval_loss += loss.item()

            _, predicted = torch.max(logits, -1)

            predict_out.extend(predicted.tolist())
            all_label_ids.extend(labels.argmax(dim=1).tolist())
            eval_accuracy = predicted.eq(labels.argmax(dim=1)).sum().item()

            total += len(labels)
            correct += eval_accuracy
            steps += 1
            if max_eval_steps is not None and steps > max_eval_steps:
                break

        f1_metrics = f1_score(
            np.array(all_label_ids).reshape(-1),
            np.array(predict_out).reshape(-1),
            average="weighted",
        )
        report = classification_report(
            np.array(all_label_ids).reshape(-1),
            np.array(predict_out).reshape(-1),
            digits=4,
        )
        print("Evaluation Report")
        print(report)

    eval_acc = correct / total
    eval_loss = eval_loss / steps

    if epoch:
        print(
            f"Evaluation at Epoch: {epoch +1 }, Acc={eval_acc}, F1={f1_metrics}, Loss={eval_loss}"
        )
    else:
        print(f"Evaluation on Test, Acc={eval_acc}, F1={f1_metrics}, Loss={eval_loss}")

    return eval_loss, eval_acc, f1_metrics


def predict(model, dataloader: DataLoader):
    # print("***** Running prediction *****")
    model.eval()
    predict_out = []

    device = next(model.parameters()).device
    with torch.no_grad():
        for char_input, word_emb, sentence_emb in dataloader:
            logits = model(
                char_input.to(device), word_emb.to(device), sentence_emb.to(device)
            )

            predicted = F.softmax(logits, dim=1)

            predict_out.extend(predicted.tolist())

    return predict_out
