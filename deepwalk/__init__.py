# -*- coding: utf-8 -*-

__author__ = 'Bryan Perozzi'
__email__ = 'bperozzi@cs.stonybrook.edu'
__version__ = '1.0.0'

import random
import graph
import walks as serialized_walks
from gensim.models import Word2Vec
from skipgram import Skipgram


def process(input, output, format="edgelist", representation_size=20, number_walks=15, walk_length=20, window_size=5, undirected=False, matfile_variable_name="network", max_memory_data_size=1000000000, seed=0, workers=1, vertex_freq_degree=False):
  if format == "adjlist":
    G = graph.load_adjacencylist(input, undirected=undirected)
  elif format == "edgelist":
    G = graph.load_edgelist(input, undirected=undirected)
  elif format == "mat":
    G = graph.load_matfile(input, variable_name=matfile_variable_name, undirected=undirected)
  else:
    raise Exception("Unknown file format: '%s'.  Valid formats: 'adjlist', 'edgelist', 'mat'" % format)

  print("Number of nodes: {}".format(len(G.nodes())))

  num_walks = len(G.nodes()) * number_walks

  print("Number of walks: {}".format(num_walks))

  data_size = num_walks * walk_length

  print("Data size (walks*length): {}".format(data_size))

  if data_size < max_memory_data_size:
    print("Walking...")
    walks = graph.build_deepwalk_corpus(G, num_paths=number_walks,
                                        path_length=walk_length, alpha=0, rand=random.Random(seed))
    print("Training...")
    model = Word2Vec(walks, size=representation_size, window=window_size, min_count=0, sg=1, hs=1, workers=workers)
  else:
    print("Data size {} is larger than limit (max-memory-data-size: {}).  Dumping walks to disk.".format(data_size, max_memory_data_size))
    print("Walking...")

    walks_filebase = output + ".walks"
    walk_files = serialized_walks.write_walks_to_disk(G, walks_filebase, num_paths=number_walks,
                                         path_length=walk_length, alpha=0, rand=random.Random(seed),
                                         num_workers=workers)

    print("Counting vertex frequency...")
    if not vertex_freq_degree:
      vertex_counts = serialized_walks.count_textfiles(walk_files, args.workers)
    else:
      # use degree distribution for frequency in tree
      vertex_counts = G.degree(nodes=G.iterkeys())

    print("Training...")
    walks_corpus = serialized_walks.WalksCorpus(walk_files)
    model = Skipgram(sentences=walks_corpus, vocabulary_counts=vertex_counts,
                     size=representation_size,
                     window=window_size, min_count=0, trim_rule=None, workers=workers)

  model.wv.save_word2vec_format(output)