import matplotlib.pyplot as plt
import numpy as np


def plot_losses_all_models(title, bert_loss_file, roberta_loss_file, sci_bert_loss_file, xlnet_loss_file):
    """
    Plots losses for all models
    :param title: title of plot
    :param bert_loss_file: filename of bert losses
    :param roberta_loss_file: filename of roberta losses
    :param sci_bert_loss_file: filename of sci ber losses
    :param xlnet_loss_file: filename of xlnet losses
    """
    bert_loss = np.load(bert_loss_file)
    roberta_loss = np.load(roberta_loss_file)
    sci_bert_loss = np.load(sci_bert_loss_file)
    xlnet_loss = np.load(xlnet_loss_file)

    fig, ax1 = plt.subplots(figsize=(16, 9))

    ax1.plot(range(len(bert_loss)), bert_loss, c='tab:red', alpha=0.25, label="BERT Train Loss")
    ax1.plot(range(len(roberta_loss)), roberta_loss, c='tab:blue', alpha=0.25, label="RoBERTA Train Loss")
    ax1.plot(range(len(sci_bert_loss)), sci_bert_loss, c='tab:green', alpha=0.25, label="Sci BERT Train Loss")
    ax1.plot(range(len(xlnet_loss)), xlnet_loss, c='tab:purple', alpha=0.25, label="XL Net Train Loss")

    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Training Loss")
    ax1.tick_params(axis='y')
    ax1.set_ylim(-0.01, 3)

    ax1.set_title(title)
    ax1.legend(loc="center right")
    plt.savefig(f"{title}.jpg")
    plt.show()


if __name__ == '__main__':
    # example
    bert_loss_file = 'bert.npy'
    roberta_loss_file = 'roberta.npy'
    sci_bert_loss_file = 'sci_bert.npy'
    xlnet_loss_file = 'xlnet.npy'
    np.save(bert_loss_file, np.array([0, 1, 2, 4, 5, 6, 8]))
    np.save(roberta_loss_file, np.array([0, 1, 1, 4, 5, 6, 8]))
    np.save(sci_bert_loss_file, np.array([1, 1, 2, 5, 5, 6, 8]))
    np.save(xlnet_loss_file, np.array([0, 0, 2, 2, 2, 2, 2]))

    plot_losses_all_models("SciNLI Training Losses", bert_loss_file, roberta_loss_file,
                           sci_bert_loss_file, xlnet_loss_file)