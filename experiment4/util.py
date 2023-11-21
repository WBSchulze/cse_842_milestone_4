import matplotlib.pyplot as plt
import numpy as np
import random
import torch


def save( epoch, model, optimizer, path = 'model.pt' ):
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict() }, path)
    

def load( model, optimizer, path = 'model.pt' ):
    checkpoint = torch.load( path )
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return epoch


def cosine_annealing( iEpoch, period ):
    epDivPeriod = ( iEpoch * 2 * np.pi ) / period
    sinusoid = 0.5 - 0.5 * np.cos( epDivPeriod )
    return sinusoid


def engineered_factor( iEpoch, period ):
    """period / 2 epochs of 0.  Period / 2 epochs rising.
       period / 2 epochs of 1.  Period should be multiple of 3."""
    partPeriod = period / 3
    normalized = iEpoch % period
    if normalized < partPeriod:
        return 0
    elif normalized < 2 * partPeriod:
        return cosine_annealing( normalized - partPeriod, 2 * partPeriod )
    else:
        return 1


def engineered_factor_switching( iEpoch, period ):
    this_factor = engineered_factor( iEpoch, period )
    last_factor = engineered_factor( iEpoch - 1, period )
    if ( ( (this_factor == 0) or (this_factor == 1) ) and (this_factor != last_factor) ):
        return True
    return False


def set_fixed_randomness(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def plot_losses(epochs, recon_losses, kl_divergences, bce_losses,
                lws_overall = None, lws_KL = None, lws_classification = None ):
    def plot_loss(epochs, loss, description, color, show_min = False, log_scale = False, lws = None):
        if show_min:
            plt.axhline( y = min( loss ), color = color, linestyle = '--' )
        if log_scale:
            plt.semilogy( epochs, loss, label=description, color=color ) 
        else:
            plt.plot(epochs, loss, label=description, color=color)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(description)
        plt.legend()
        if lws is not None:
            ax2 = plt.twinx()
            ax2.plot( epochs, lws, color = '#cccccc')

    plt.figure(figsize=(10, 8))

    plt.subplot(2, 1, 1)
    plot_loss(epochs, recon_losses, 'Reconstruction Binary Cross-Entropy Loss', 'blue', 
              show_min = True, log_scale = True, lws = lws_overall )
        
    plt.subplot(2, 2, 3)
    plot_loss(epochs, kl_divergences, 'KL Divergence Loss', 'red', 
              log_scale = True, lws = lws_KL )
    
    plt.subplot(2, 2, 4)
    plot_loss(epochs, bce_losses, 'Classification Binary Cross-Entropy Loss', 'green', 
              log_scale = True, lws = lws_classification )

    plt.tight_layout()
    plt.savefig(f'losses.png')
    plt.close()
