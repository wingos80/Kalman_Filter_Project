import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def plotter(ax, x, y, ylabel, colors=None, log=False):
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

    i=0
    for key, value in y.items():
        ax.plot(x, value[0], color=colors[i] if colors else f'C{i}',label=key, alpha=value[1])
        i+=1
    if log:
        ax.set_yscale('log')
    ax.grid(True); ax.set_ylabel(ylabel); ax.legend()
    

def make_plots(x, ys, title, xlabel, ylabel, colors=None, save=False, log=None):
    """
    Make multiple plots in one figure    
    Parameters
    ----------    
    x : np.array
        x-axis vector    
    ys : list
        List of dictionaries, each dictionary is a plot        
    title : string
        Title of the plot        
    xlabel : string
        Label of the x-axis        
    ylabel : list
        List of labels for the y-axis of each subplots    
    colors : list, optional
        List of colors for each plot, by default None
    save : bool, optional
        Save the figure to a file, by default False        
    log : int, optional
        Index of the plot to be plotted in log scale, by default None    
    """
    fig, ax = plt.subplots(len(ys), 1, sharex=True)
    fig.canvas.manager.set_window_title(title) 
    plt.subplots_adjust(top=0.965, bottom=0.110, right=0.990, left=0.165)
    if len(ylabel) != len(ys):
        print('---Plot Warning, Number of ylabel must be equal to number of plots')
            

    if len(ys) == 1:
        ax = [ax]
    
    ax[-1].set_xlabel(xlabel)

    for i, y in enumerate(ys):
        if i == log:
            plotter(ax[i], x, y, ylabel[i], colors=colors,log=True)
        else:
            plotter(ax[i], x, y, ylabel[i], colors=colors,log=None)
    
    if save:
        plt.savefig(f'{title.replace(" ", "_").replace("-", "_")}.pdf')
