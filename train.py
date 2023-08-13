from sklearn.metrics import accuracy_score, f1_score
import torch
from tqdm import tqdm

def train_model(model, train_L, valid_L, loss_criterion, optim, NUM_EPOCHS, DEVICE):
    model = model.to(DEVICE)

    # Initialize metrics parameters
    best_acc = 0.0
    best_model = None

    final_train_losses = []
    final_valid_losses = []
    final_train_accs = []
    final_valid_accs = []
    final_train_f1 = []
    final_valid_f1 = []
    train_losses = []
    valid_losses = []
    train_acc = []
    valid_acc = []
    train_f1 = []
    valid_f1 = []


    # Train the network
    for epoch in range(NUM_EPOCHS):
        model.train()
        for features, labels in tqdm(train_L, desc=f"(train) Epoch [{epoch+1}/{NUM_EPOCHS}]"):
            features = features.to(DEVICE).float()
            # features = features.float()
            labels = labels.to(DEVICE)
            # labels = labels
            optim.zero_grad()
            outputs = model(features)
            loss = loss_criterion(outputs, labels)
            loss.backward()
            optim.step()
            train_losses.append(loss.item())

            pred = outputs.argmax(dim=1)
            acc = accuracy_score(labels.cpu().numpy(), pred.cpu().numpy())
            f1 = f1_score(labels.cpu().numpy(), pred.cpu().numpy(), average='macro')
            train_acc.append(acc)
            train_f1.append(f1)

        model.eval()
        with torch.no_grad():
            for features, labels in tqdm(valid_L, desc=f"(valid) Epoch [{epoch+1}/{NUM_EPOCHS}]"):
                features = features.to(DEVICE).float()
                # features = features.float()
                labels = labels.to(DEVICE)
                # labels = labels
                # torch.cuda.synchronize()
                outputs = model(features)
                loss = loss_criterion(outputs, labels)
                valid_losses.append(loss.item())

                pred = outputs.argmax(dim=1)
                acc = accuracy_score(labels.cpu().numpy(), pred.cpu().numpy())
                f1 = f1_score(labels.cpu().numpy(), pred.cpu().numpy(), average='macro')
                valid_acc.append(acc)
                valid_f1.append(f1)

            train_loss = sum(train_losses) / len(train_losses)
            train_accuracy = sum(train_acc) / len(train_acc)
            train_f1_score = sum(train_f1) / len(train_f1)
    
            valid_loss = sum(valid_losses) / len(valid_losses)
            valid_accuracy = sum(valid_acc) / len(valid_acc)
            valid_f1_score = sum(valid_f1) / len(valid_f1)

            final_train_losses.append(train_loss)
            final_valid_losses.append(valid_loss)

            final_train_accs.append(train_accuracy)
            final_valid_accs.append(valid_accuracy)

            final_train_f1.append(train_f1_score)
            final_valid_f1.append(valid_f1_score)

            if valid_accuracy > best_acc:
                best_acc = valid_accuracy
                best_model = model.state_dict()

            print('Train Loss: {:.4f}, Train Accuracy: {:.4f}, Valid Loss: {:.4f}, Valid Accuracy: {:.4f}'
                .format(train_loss, train_accuracy, valid_loss, valid_accuracy))

    losses = [final_train_losses, final_valid_losses]
    accuracies = [final_train_accs, final_valid_accs]
    f1_scores = [final_train_f1, final_valid_f1]
    print('Finished Training\n')
    return best_model, losses, accuracies, f1_scores
