import matplotlib.pyplot as plt
import pandas as pd

def performance(model, best_model, n_classes, losses, accuracies, f1_scores, task):
    model.load_state_dict(best_model) 

    # plot training and validation losses
    fig, ax = plt.subplots()
    ax.plot(losses[0], label='Train Loss')
    ax.plot(losses[1], label='Valid Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss for {task}')
    plt.legend()
    plt.savefig('results/plots/training_validation_loss_{}.png'.format(task))
    plt.close(fig)

    # plot accuracies
    fig, ax = plt.subplots()
    ax.plot(accuracies[0], label='Train Accuracy')
    ax.plot(accuracies[1], label='Valid Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Training and Validation Accuracy for {task}')
    best_train_acc = max(accuracies[0])
    best_valid_acc =  max(accuracies[1])
    # plt.text(0.5, -0.2, f'Best Train F1-Score: {best_train_acc:.2f}, Best Val F1-Score: {best_valid_acc:.2f}', ha='center', transform=ax.transAxes)
    plt.legend()
    plt.savefig('results/plots/training_validation_accuracy_{}.png'.format(task))
    plt.close(fig)

    # plot f1-score
    fig, ax = plt.subplots()
    ax.plot(f1_scores[0], label='Train F1-Score')
    ax.plot(f1_scores[1], label='Valid F1-Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1-Score')
    plt.title(f'Training and Validation F1-Score for {task}')
    best_train_f1 = max(f1_scores[0])
    best_valid_f1 =  max(f1_scores[1])
    # plt.text(0.5, -0.2, f'Best Train F1-Score: {best_train_f1:.2f}, Best Val F1-Score: {best_valid_f1:.2f}', ha='center', transform=ax.transAxes)
    plt.legend()
    plt.savefig('results/plots/training_validation_f1_score_{}.png'.format(task))
    plt.close(fig)

    perf_df = pd.DataFrame([[best_train_acc, best_valid_acc, best_train_f1, best_valid_f1]], 
                            columns=['train_acc', 'val_acc', 'train_f1', 'val_f1'])

    return perf_df


