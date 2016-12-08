# visualization imports
import matplotlib.pyplot as plt
#import matplotlib.colors as colors
import seaborn as sb
import pandas


def create_heatmap(plot_data, title_name, plot_path):
    print("Creating heatmap...")
    # prevent showing figure after saving for more automated use
    plt.ioff()
    sb.set(context = "paper", font = "monospace")
    figure, axes = plt.subplots(figsize = (12,9))
    axes.set_title(title_name, fontsize = 14)
    sb.heatmap(plot_data, cmap = 'inferno') # maybe black to red
    figure.tight_layout()
    plt.xticks(rotation = "vertical")
    plt.yticks(rotation = "horizontal")
    figure.savefig(plot_path, bbox_inches = "tight")
    plt.close(figure)

title = ["Simulated Data with 10 Means - No Dropout", "Simulated Data with 10 Means - Dropout", "Simulated Data with 5 Means - Dropout",
           "Simulated Data with 5 Means - No Dropout", "?", "Simulated Data with 10 Means - No Dropout",
           "Simulated Data with 10 Means - No Dropout", "Simulated Data with 2 Means - Dropout", "Simulated Data with 2 Means - No Dropout",
           "Simulated Data with 5 Means - Dropout", "Simulated Data with 5 Means - No Dropout"]

csv_files = ['TestSet2Sample.csv','10NDSample2.csv', '5Sample2.csv','5NDSample2.csv','set2cluster.csv','TestSet10SampleNoDrop.csv','10Sample2.csv','TestSet10Sample.csv','TestSet5SampleNoDrop.csv','TestSet5Sample.csv','TestSet2SampleNoDrop.csv']

save_files = ['data/TestSet2Sample_heatmap.png',
'data/10NDSample2_heatmap.png',
'data/5Sample2_heatmap.png',
'data/5NDSample2_heatmap.png',
'data/set2cluster_heatmap.png',
'data/TestSet10SampleNoDrop_heatmap.png',
'data/10Sample2_heatmap.png',
'data/TestSet10Sample_heatmap.png',
'data/TestSet5SampleNoDrop_heatmap.png', 'data/TestSet5Sample_heatmap.png',
'data/TestSet2SampleNoDrop_heatmap.png']

csv_files = ["data/10Sample2.csv", "data/10NDSample2.csv"]
title = ["Simulated Data with 10 Means - Dropout", "Simulated Data with 10 Means - No Dropout"]
save_files = ["plots/10Sample2_heatmap.png", "plots/10NDSample2_heatmap.png"]


print(len(csv_files), len(save_files), len(title))
for i in range(len(csv_files)):
    print("Running {0}".format(csv_files[i]))
    data = pandas.read_csv(csv_files[i], index_col = 0, header = 0)
    create_heatmap(data, title[i], save_files[i])
