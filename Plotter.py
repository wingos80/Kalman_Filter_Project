import matplotlib.pyplot as plt

def plotter(x, y, title, xlabel, ylabel, save=False):
    """
    Plotter function for general graphs

    Parameters
    ----------
    x : np.array
        x-axis vector

    y : dictionary
        List of length 2 in each value, 1st element is f(x),
        2nd element is transparency(alpha) of that graph,
        each key is the label for the corresponding graph

    title : string
        Title of the plot
    
    xlabel : string
        Label of the x-axis
    
    ylabel : string
        Label of the y-axis

    save : bool, optional
        Save the figure to a file, by default False        
    """

    fig = plt.figure(figsize=(12,6))
    ax  = fig.add_subplot(1, 1 ,1 )
    for key, value in y.items():
        ax.plot(x, value[0], label=key, alpha=value[1])
    plt.grid(True)
    plt.title(title, fontsize = 18); plt.xlabel(xlabel); plt.ylabel(ylabel); plt.legend()
    
    if save:
        plt.savefig(title + '.png')