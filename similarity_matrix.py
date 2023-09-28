import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.special import softmax
from datetime import datetime as dt
import argparse
import matplotlib.pyplot as plt
from seaborn import heatmap
import os,sys

from Data_loader.EMData import read_data_df

def add_args(parser):
    parser.add_argument('particles', type=os.path.abspath, help='Input particles path (.star)')
    parser.add_argument('--input_emb', type=os.path.abspath, help='The place the embedding numpy file')

    return parser

def main(args):
    numpy_path = args.input_emb
    meta_path = args.particles
    all_embeddings = np.load(numpy_path, allow_pickle=True)
    #print(all_embeddings.shape)
    all_particles_meta = read_data_df(meta_path)
    dataframe = all_particles_meta.star2dataframe(relion31=True)
    helicaldic, filament_index = all_particles_meta.extract_helical_select(dataframe)
    corpus_ignore = all_particles_meta.filament_index(helicaldic)
    all_matrix = []
    filament_length = []
    for i in range(len(corpus_ignore)):
        select_particles = corpus_ignore[i]
        list = all_embeddings[select_particles]
        matrix_condense = pdist(list)
        matrix = -squareform(matrix_condense)
        matrix_softmax = softmax(matrix, axis=-1)
        np.fill_diagonal(matrix_softmax, 0)
        #matrix_softmax_new = softmax(matrix_softmax, axis=-1)
        all_matrix.append(matrix_softmax)
        filament_length.append(len(all_matrix))

    idx = np.argmax(filament_length)
    select_matrix = all_matrix[0]
    heatmap(select_matrix, annot=False)
    a = select_matrix[0]
    print(a.sum())
    plt.show()
    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    main(add_args(parser).parse_args())