
import matplotlib.pyplot as plt

from matplotlib.colors import Normalize
import matplotlib.cm as cm

import json
import numpy as np
from PIL import Image
from operator import itemgetter
#### Plotting
def seeds_json_to_plot_ready(json_path):
    

    with open(json_path, 'r') as log_file:
        data = json.load(log_file)

    plot_data = {}
    for c in data:
        plot_data[c['threshold']] = {}
        plot_data[c['threshold']]['Volume'] = c["Whole Volume"]
        plot_data[c['threshold']]['footprints'] = c["footprints"]
        seeds = c['seeds']
        for seed in seeds:
            ero_iter = seed['ero_iter']
            size_ccomp = seed['size_ccomp']
            x = seed['number_of_ccomps']
            plot_data[c['threshold']][f'erosion_{ero_iter}'] = {'x': list(range(1,x+1)), 'y': size_ccomp}
            
    return plot_data

def plot_seeds_log(plot_data, fig_path):  
    
    # Define a list of 5 colors
    colors = ['red', 'blue', 'green', 'orange', 'purple']

    # Define a list of 5 different shapes (markers)
    shapes = ['o', 's', '^', 'D', 'P']  # circle, square, triangle, diamond, pentagon
    # Create subplots (one for each cate1)

    fig_width = 6
    fig, axes = plt.subplots(1, len(plot_data), 
                            figsize=(fig_width*len(plot_data), fig_width), 
                            sharey=True)
    # Plot each group
    for idx_thre, (thre, ero_data) in enumerate(sorted(plot_data.items())):
        ax = axes[idx_thre]  # Select the subplot
        ax.set_title(f'Thre: {thre}')
        footprints = ero_data["footprints"]
        
        ax.axhline(y= ero_data["Volume"], color='red', linestyle='--', linewidth=2)
        for idx_ero, (ero, value) in enumerate(ero_data.items()):
            if ero!="Volume" and ero!="footprints":
                x = value['x']
                y = value['y']
            
                # Plot the points
                ax.scatter(x, y, color=colors[idx_ero % len(colors)], 
                            marker=shapes[idx_ero % len(colors)],
                            label=f"Thre:{thre}, Ero: {ero}\n{footprints}")
                
                # Plot the line segments
                ax.plot(x, y, color=colors[idx_ero % len(colors)])
        
                
        
        ax.set_xlabel('Component id')
        ax.legend()

    axes[0].set_ylabel('Volume')

    # Save the plot to a file
    plt.savefig(fig_path)

    # Close the plot to free memory and avoid displaying it
    plt.close()

def plot_seeds_log_json(data, fig_path):
    # Define a list of 5 colors
    colors = ['red', 'blue', 'green', 'orange', 'purple']

    # Define a list of 5 different shapes (markers)
    shapes = ['o', 's', '^', 'D', 'P']  # circle, square, triangle, diamond, pentagon
    # Create subplots (one for each cate1)

    data = sorted(data, key=itemgetter('threshold'))
    
    unique_list = []
    seen_thresholds = set()

    for d in data:
        threshold = d['threshold']
        if threshold not in seen_thresholds:
            unique_list.append(d)
            seen_thresholds.add(threshold)

    data = unique_list

    n_thres = len(data)

    fig_width = 6
    fig, axes = plt.subplots(1, n_thres, 
                            figsize=(fig_width*n_thres, fig_width), 
                            sharey=True)


    for idx_thre, data_entry in enumerate(data):
        threshold = data_entry['threshold']
        volume = data_entry["Whole Volume"]
        footprints = data_entry["footprints"]
        seeds = data_entry['seeds']
        
        ax = axes[idx_thre]  # Select the subplot
        ax.set_title(f'Thre: {threshold}')
        ax.axhline(y= volume, color='red', linestyle='--', linewidth=2)
        
        for idx_ero,seed in enumerate(seeds):
            ero_iter = seed['ero_iter']
            size_ccomp = seed['size_ccomp']
            x = list(range(1,seed['number_of_ccomps']+1))
            y = seed['size_ccomp']
            
            # Plot the points
            ax.scatter(x, y, color=colors[idx_ero % len(colors)], 
                        marker=shapes[idx_ero % len(colors)],
                        label=f"Thre:{threshold}, Ero: {ero_iter}\n{footprints}")
            
            # Plot the line segments
            ax.plot(x, y, color=colors[idx_ero % len(colors)])
        
                
        
        ax.set_xlabel('Component id')
        ax.legend()

    axes[0].set_ylabel('Volume')
    
    # Save the plot to a file
    plt.savefig(fig_path)

    # Close the plot to free memory and avoid displaying it
    plt.close()


def merge_plots(plot_list, file_path):
    # Open all images and convert them to numpy arrays
    images = [np.array(Image.open(image_path)) for image_path in plot_list]

    # Vertically stack the images using np.vstack
    stacked_images = np.vstack(images)

    # Convert the result back to an image
    result_image = Image.fromarray(stacked_images)
    
    result_image.save(file_path)
    
    
def plot_grow(df, fig_path):

    thresholds = df['cur_threshold'].unique()
    norm = Normalize(0, len(thresholds))
    cmap = cm.viridis
    # colors = cmap(norm(df['cur_threshold'].unique()))
    
    colors =cmap(norm(list(range(len(thresholds)))))
    # Create a line plot with id as x, value as y, and threshold as color
    plt.figure(figsize=(20, 10))


    # Plot the line with different colors based on the threshold
    for i, threshold in enumerate(df['cur_threshold'].unique()):
        subset = df[df['cur_threshold'] == threshold]
        plt.plot(subset['id'], subset['grow_size'], color = colors[i], label=f'Threshold {threshold}')
        
        plt.scatter(subset['id'], subset['grow_size'], color = colors[i])
        
        plt.axhline(y=subset['full_size'].unique(),
                    linestyle='--', 
                    color = colors[i])

    # Customize x-ticks to include both id and name
    xticks_labels = [f'{id_}: dilate:{dilate_step_}. threshold: {threshold_}' for id_, dilate_step_,threshold_ 
                    in zip(df['id'], df['cur_dilate_step'], df['cur_threshold'])]
    plt.xticks(df['id'], xticks_labels)


    # Add labels and legend
    plt.xlabel('ID and Filename')
    plt.ylabel('Value')
    plt.title('Grow size vs threhold and dilation')
    plt.legend(title='Threshold')

    # Save the plot to a file
    plt.savefig(fig_path)

    # Close the plot to free memory and avoid displaying it
    plt.close()
