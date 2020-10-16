import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def gen_heatmap(csv_path, csv_label, csv_label_len, output_img):
    dfcsv = pd.read_csv(csv_path, header=None)
    plt.figure(figsize=(15,9))
    sns.heatmap(dfcsv,cmap="Greens")
    plt.xticks(np.arange(0.5, csv_label_len+0.5, 1), csv_label, rotation=270)
    plt.yticks(np.arange(0.5, csv_label_len+0.5, 1), csv_label, rotation='horizontal')
    plt.tight_layout()
    plt.savefig('/home/ruiliu/Development/mtml-tf/hyperband/'+output_img,format='pdf')


if __name__ == "__main__":
    #csvlabel=('(mlp,32,Adam,1,relu)','(mlp,32,SGD,1,relu)','(mlp,128,Adam,1,relu)','(mlp,128,SGD,1,relu)','(mobilenet,32,Adam)','(mobilenet,32,SGD)','(mobilenet,128,Adam)','(mobilenet,128,SGD)','(resnet,32,Adam)','(resnet,32,SGD)','(resnet,128,Adam)','(resnet,128,SGD)','(densenet,32,Adam)','(densenet,32,SGD)','(densenet, 128,Adam)','(densenet,128,SGD)')
    
    #csvlabel=('(mlp,32,Adam,1,sigmoid)','(mlp,32,Adam,1,relu)','(mlp,32,Adam,3,sigmoid)','(mlp,32,Adam,3,relu)','(mlp,32,SGD,1,sigmoid)','(mlp,32,SGD,1,relu)','(mlp,32,SGD,3,sigmoid)','(mlp,32,SGD,3,relu)','(mlp,50,Adam,1,sigmoid)','(mlp,50,Adam,1,relu)','(mlp,50,Adam,3,sigmoid)','(mlp,50,Adam,3,relu)','(mlp,50,SGD,1,sigmoid)','(mlp,50,SGD,1,relu)','(mlp,50,SGD,3,sigmoid)','(mlp,50,SGD,3,relu)')

    #csvlabel=('(mlp,32,Adam,1,sigmoid)','(mlp,32,Adam,1,relu)','(mlp,32,Adam,3,sigmoid)','(mlp,32,Adam,3,relu)','(mlp,32,SGD,1,sigmoid)','(mlp,32,SGD,1,relu)','(mlp,32,SGD,3,sigmoid)','(mlp,32,SGD,3,relu)','(mlp,50,Adam,1,sigmoid)','(mlp,50,Adam,1,relu)','(mlp,50,Adam,3,sigmoid)','(mlp,50,Adam,3,relu)','(mlp,50,SGD,1,sigmoid)','(mlp,50,SGD,1,relu)','(mlp,50,SGD,3,sigmoid)','(mlp,50,SGD,3,relu)','(mobilenet,32,Adam)','(mobilenet,32,Adam)','(mobilenet,32,Adam)','(mobilenet,32,Adam)','(mobilenet,32,SGD)','(mobilenet,32,SGD)','(mobilenet,32,SGD)','(mobilenet,32,SGD)','(mobilenet,50,Adam)','(mobilenet,50,Adam)','(mobilenet,50,Adam)','(mobilenet,50,Adam)','(mobilenet,50,SGD)','(mobilenet,50,SGD)','(mobilenet,50,SGD)','(mobilenet,50,SGD)')

    #(mlp,32,Adam),(mlp,32,SGD),(mlp,32,Adagrad),(mlp,32,Momentum),(mlp,64,Adam),(mlp,64,SGD),(mlp,64,Adagrad),(mlp,64,Momentum),(mlp,128,Adam),(mlp,128,SGD),(mlp,128,Adagrad),(mlp,128,Momentum),(mobilenet,32,Adam),(mobilenet,32,SGD),(mobilenet,32,Adagrad),(mobilenet,32,Momentum),(mobilenet,64,Adam),(mobilenet,64,SGD),(mobilenet,64,Adagrad),(mobilenet,64,Momentum),(mobilenet,128,Adam),(mobilenet,128,SGD),(mobilenet,128,Adagrad),(mobilenet,128,Momentum),(resnet,32,Adam),(resnet,32,SGD),(resnet,32,Adagrad),(resnet,32,Momentum),(resnet,64,Adam),(resnet,64,SGD),(resnet,64,Adagrad),(resnet,64,Momentum),(resnet,128,Adam),(resnet,128,SGD),(resnet,128,Adagrad),(resnet,128,Momentum),(densenet,32,Adam),(densenet,32,SGD),(densenet,32,Adagrad),(densenet,32,Momentum),(densenet,64,Adam),(densenet,64,SGD),(densenet,64,Adagrad),(densenet,64,Momentum),(densenet,128,Adam),(densenet,128,SGD),(densenet,128,Adagrad),(densenet,128,Momentum)

    csvlabel=('(mlp,32,Adam)','(mlp,32,SGD)','(mlp,32,Adagrad)','(mlp,32,Momentum)','(mlp,64,Adam)','(mlp,64,SGD)','(mlp,64,Adagrad)','(mlp,64,Momentum)','(mlp,128,Adam)','(mlp,128,SGD)','(mlp,128,Adagrad)','(mlp,128,Momentum)','(mobilenet,32,Adam)','(mobilenet,32,SGD)','(mobilenet,32,Adagrad)','(mobilenet,32,Momentum)','(mobilenet,64,Adam)','(mobilenet,64,SGD)','(mobilenet,64,Adagrad)','(mobilenet,64,Momentum)','(mobilenet,128,Adam)','(mobilenet,128,SGD)','(mobilenet,128,Adagrad)','(mobilenet,128,Momentum)','(resnet,32,Adam)','(resnet,32,SGD)','(resnet,32,Adagrad)','(resnet,32,Momentum)','(resnet,64,Adam)','(resnet,64,SGD)','(resnet,64,Adagrad)','(resnet,64,Momentum)', '(resnet,128,Adam)','(resnet,128,SGD)','(resnet,128,Adagrad)','(resnet,128,Momentum)','(densenet,32,Adam)','(densenet,32,SGD)','(densenet,32,Adagrad)','(densenet,32,Momentum)','(densenet,64,Adam)','(densenet,64,SGD)','(densenet,64,Adagrad)','(densenet,64,Momentum)','(densenet,128,Adam)','(densenet,128,SGD)','(densenet,128,Adagrad)','(densenet,128,Momentum)')

    csv_path = 'pairwise-exp4.csv'
    output_img = 'heatmap4.pdf'
    gen_heatmap(csv_path, csvlabel, 48, output_img)