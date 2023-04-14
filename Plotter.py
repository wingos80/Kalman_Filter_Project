import matplotlib.pyplot as plt
import numpy as np

def plotter(ax, x, y, ylabel, log=False):
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

    # fig = plt.figure(figsize=(12,6))
    # ax  = fig.add_subplot()
    for key, value in y.items():
        ax.plot(x, value[0], label=key, alpha=value[1])
    if log:
        ax.set_yscale('log')
    ax.grid(True); ax.set_ylabel(ylabel); ax.legend()
    

def make_plots(x, ys, title, xlabel, ylabel, save=False, log=None):
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
    
    save : bool, optional
        Save the figure to a file, by default False
        
    log : int, optional
        Index of the plot to be plotted in log scale, by default None    
    """
    fig, ax = plt.subplots(len(ys), 1, figsize=(12, 6), sharex=True)
    fig.suptitle(title, fontsize = 18);
    
    if len(ylabel) != len(ys):
        print('---Plot Warning, Number of ylabel must be equal to number of plots')
            

    if len(ys) == 1:
        ax = [ax]
    
    ax[-1].set_xlabel(xlabel)

    for i, y in enumerate(ys):
        if i == log:
            plotter(ax[i], x, y, ylabel[i], log=True)
        else:
            plotter(ax[i], x, y, ylabel[i], log=None)
    
    if save:
        plt.savefig(title + '.png')

# x = np.linspace(0, 10, 100)
# y = {'f(x)': [np.sin(x), 1], 'g(x)': [np.cos(x), 0.5]}
# y2 = {'f(x)': [np.tan(x), 1], 'g(x)': [np.cos(x), 0.5], 'h(x)': [np.sin(x) + np.cos(x), 0.5]}

# make_plots(x, [y], 'Sine and Cosine', 'x', ['f(x), g(x)', 'sdfg'], save=False, log=None)
# plt.show()