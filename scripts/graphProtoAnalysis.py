import uproot
import numpy as np
import graph_nets as gn
import matplotlib.pyplot as plt
import matplotlib as mpl
import networkx as nx
from collections import OrderedDict
import pandas as pd
from sklearn.neighbors import NearestNeighbors


# TODO: understand why plotting is so time-expensive!!!

def check(nevts, files):
  '''check that files can be opened correctly and that there are enough entries'''
  tot_entries = 0

  for fn in files:
    try:
      tot_entries += uproot.open(fn)['recosimdumper']['caloTree'].numentries
    except:
      print('file f={} may not exist or does not have tree=recosimdumper/caloTree'.format(fn))

  if tot_entries < nevts:
    print('Chosen number of events is too large, using max_nevts=', tot_entries)
    return tot_entries
 
  else:
    return nevts 

def getDataFrames(nevts,files):
  '''creates the dataframes for sim and reco, one per event...''' 
  simdfs = []
  recodfs = []

#  # wrong way of reading arrays I would say... 
#  for inev in range(0,nevts):
#    simdf = pd.DataFrame({ 'energy': uproot.lazyarray(files, 'recosimdumper/caloTree', 'simHit_energy')[inev],
#                           'ieta'  : uproot.lazyarray(files, 'recosimdumper/caloTree', 'simHit_ieta')[inev],
#                           'iphi'  : uproot.lazyarray(files, 'recosimdumper/caloTree', 'simHit_iphi')[inev],
#                           'icalo' : uproot.lazyarray(files, 'recosimdumper/caloTree', 'simHit_icP')[inev],
#                         })  
#    simdfs.append(simdf)
#
#    recodf = pd.DataFrame({'energy': uproot.lazyarray(files, 'recosimdumper/caloTree', 'pfRecHit_energy')[inev],
#                           'ieta'  : uproot.lazyarray(files, 'recosimdumper/caloTree', 'pfRecHit_ieta')[inev],
#                           'iphi'  : uproot.lazyarray(files, 'recosimdumper/caloTree', 'pfRecHit_iphi')[inev],
#                           })
#    recodfs.append(recodf)

  events = uproot.open(files[0])['recosimdumper']['caloTree'] # only take the first one
  for inev in range(0,nevts):
    simdf = pd.DataFrame({ 'energy': events.array('simHit_energy')[inev],
                           'ieta': events.array('simHit_ieta')[inev], 
                           'iphi': events.array('simHit_iphi')[inev], 
                           'icalo': events.array('simHit_icP')[inev], 
                         })  
    simdfs.append(simdf)

    recodf = pd.DataFrame({'energy': events.array('pfRecHit_energy')[inev],
                           'ieta': events.array('pfRecHit_ieta')[inev],
                           'iphi': events.array('pfRecHit_iphi')[inev],
                           })
    recodfs.append(recodf)

  return simdfs,recodfs

def getDictsFromDFs(recodfs,simdfs,doSelection=False,matricesSel=[]):
  '''
  Starting from reco and sim information in pandas dataframes, creates a graph structure for reco and true clusters, with the format specified below,
  optionally performs a selection of possible edges, based on a matrix

     nodes_0 = [[10.0, 2, 3],  # Node 0  # energy, ieta, iphi
                [11.3, 0, 2],  # Node 1
                [34.3, 2, 1],  # Node 2
                [13.0, 3, 3],  # Node 3
                [14.0, 4, 4]]  # Node 4 
     
      edges_0 = [[1.],  # Edge 0  # probability that this is a true edge, equals 1. for every edge in a reco graph, equals 0. or 1. for true graph (target value)
              [1.],  # Edge 1
              [0.],  # Edge 2
              [1.],  # Edge 3
              [0.],  # Edge 4
              [1.]]  # Edge 5

     senders_0 = [0,  # Index of the sender node for edge 0
             1,  # Index of the sender node for edge 1
             1,  # Index of the sender node for edge 2
             2,  # Index of the sender node for edge 3
             2,  # Index of the sender node for edge 4
             3]  # Index of the sender node for edge 5

     receivers_0 = [1,  # Index of the receiver node for edge 0
               2,  # Index of the receiver node for edge 1
               3,  # Index of the receiver node for edge 2
               0,  # Index of the receiver node for edge 3
               3,  # Index of the receiver node for edge 4
               4]  # Index of the receiver node for edge 5 
  '''

  reco_dicts = []
  true_dicts = []

  assert(len(recodfs)==len(simdfs))

  for inev,recodf in enumerate(recodfs):
    simdf = simdfs[inev]

    #define nodes, edges and globals
    nodes = []
    edges = []
    true_edges = []
    senders = []
    receivers = []
    #globals = []

    # straight conversion from pd to list of lists
    nodes = recodf.values.tolist() 
    
    # create edges between every node with every other node, in both directions
    for i,inode in recodf.iterrows():
      for j,jnode in recodf.iterrows():
        
        # if true, do KNN selection on the reco graph, i.e. only consider edges between nearest neighbours
        if doSelection: 
          #print('debug matrix =', matricesSel[inev][(i,j)])
          if matricesSel[inev][(i,j)]==0: continue 
        
        # do not create a self-edge
        if i==j: continue 

        ## set edges for reco graph
        score = np.random.rand(1).tolist()
        edges.append(score) # []
        senders.append(i)
        receivers.append(j)

        ## set edges for true graph
        # find indices of i,j reco ndoes in the sim df 
        index_i_sim = simdf[(simdf['ieta'] == inode['ieta']) & (simdf['iphi'] == inode['iphi'])].index.values
        index_j_sim = simdf[(simdf['ieta'] == jnode['ieta']) & (simdf['iphi'] == jnode['iphi'])].index.values

        true_score = [0.]
        if len(index_i_sim)==1 and len(index_j_sim)==1:
          if simdf.at[index_i_sim[0], 'icalo']==simdf.at[index_j_sim[0], 'icalo']: # if this edge connects two hits from the same caloparticle
            true_score = [1.]
        elif len(index_i_sim)>1 or len(index_j_sim)>1:
          print('Warning: there seems to be more than one node in the sim graph with the same coordinates... does not match expectation...')
          print(index_i_sim,index_j_sim)
        else:
          print('One of the nodes does not exist in the sim graph, setting true edge score to 0.')

        true_edges.append(true_score) # []

    reco_dict = {
      'globals': None,
      'nodes': nodes,
      'edges': edges,
      'senders': senders,
      'receivers': receivers,
    }
    
    true_dict = {
      'globals': None,
      'nodes': nodes,
      'edges': true_edges,  # the only difference is the score of the edges
      'senders': senders,
      'receivers': receivers,
    }

    reco_dicts.append(reco_dict)
    true_dicts.append(true_dict)

  return reco_dicts,true_dicts

def pruneDFs(dfs):
  '''return a list of dataframes where each is pruned with some specified selection criteria'''
  new_dfs = []
  for df in dfs:
    new_dfs.append(df[df['energy'] > 0.01].reset_index().drop('index', axis=1)) 
    # do a selection on initial df, reset the index, drop the old index column
  return new_dfs

def plotXY(dfs,what='reco'):
 ''' plot the ieta and iphi of the rechits of the event'''
 for iev,df in enumerate(dfs): 
  print('===> In plotXY(), evt={}'.format(iev))
  plt.figure()
  plt.ioff()
  plt.hist2d(x=df.ieta, y=df.iphi, bins=[180,360], range=[[-90,90],[0,360]], weights=df.energy, norm=mpl.colors.LogNorm(), vmin=0.1, vmax=100)
  plt.colorbar().set_label('Energy (GeV)')
  plt.xlabel('i$\eta$')
  plt.ylabel('i$\phi$')
  plt.title('{w} hits, event {i}'.format(i=iev, w=what))
  plt.grid(True, which='both')
  plt.savefig('plots/{w}_event{i}.pdf'.format(i=iev, w=what))
  plt.savefig('plots/{w}_event{i}.png'.format(i=iev, w=what))
  plt.close()


def plot_graph_nx(nx_graph, ax, pos=None):
  '''
  plot single netowrkx graph
  '''
  #node_labels = {node: "{:.3g}".format(data["features"][0])
  #               for node, data in graph.nodes(data=True)
  #               if data["features"] is not None}
  edge_labels = {(sender, receiver): "{:.3g}".format(data["features"][0])
                 for sender, receiver, data in nx_graph.edges(data=True)
                 if data["features"] is not None}
  #global_label = ("{:.3g}".format(graph.graph["features"][0])
  #                if graph.graph["features"] is not None else None)

  if pos is None:  
    for i in range(0,len(nx_graph)):
      nx_graph.add_node(i, pos=list((nx_graph.nodes[i]['features'][1],nx_graph.nodes[i]['features'][2])))  # these represent the ieta,iphi coordinates!
    pos = nx.get_node_attributes(nx_graph, "pos")
    
  #nx.draw_networkx(nx_graph, pos, ax=ax, with_labels=False, node_color="r", node_size=2) # labels=node_labels
  nx.draw(nx_graph, pos, ax=ax, node_color="r", node_size=2, with_labels=False)
  nx.draw_networkx_edge_labels(nx_graph, pos, edge_labels, font_size=5, bbox=dict(boxstyle='round',facecolor = 'white',edgecolor='white',alpha=0.5), ax=ax, node_color="r", node_size=2)

  #if global_label:
  #  plt.text(0.05, 0.95, global_label, transform=ax.transAxes)

  ax.yaxis.set_visible(False)
  ax.xaxis.set_visible(False)
  return pos

def plot_graphs_tuple_np(graphs_tuple,label):
  '''
  plot `graphs.GraphsTuple` by converting to networkx format
  '''
  networkx_graphs = gn.utils_np.graphs_tuple_to_networkxs(graphs_tuple)
  num_graphs = len(networkx_graphs)
  fig, axes = plt.subplots(1, num_graphs, figsize=(5*num_graphs, 5))
  if num_graphs == 1:
    axes = axes,
  for nx_graph, ax in zip(networkx_graphs, axes): 
    plot_graph_nx(nx_graph, ax, pos=None)
  fig.savefig('./plots/graphs_{}.pdf'.format(label))
  fig.savefig('./plots/graphs_{}.png'.format(label))

def plot_compare_graphs(graphs_tuples, labels):
  '''
  plot first graph of each `graphs.GraphsTuple`
  '''
  pos = None
  num_graphs = len(graphs_tuples)
  fig, axes = plt.subplots(1, num_graphs, figsize=(5*num_graphs, 5))
  if num_graphs == 1:
    axes = axes,
  pos = None
  for name, graphs_tuple, ax in zip(labels, graphs_tuples, axes):
    graph = gn.utils_np.graphs_tuple_to_networkxs(graphs_tuple)[0]   
    pos = plot_graph_nx(graph, ax, pos=pos)
    ax.set_title(name)
  
  fig.savefig('./plots/graphs_{}.pdf'.format('_'.join(labels)))
  fig.savefig('./plots/graphs_{}.png'.format('_'.join(labels)))

def plotGraphs(graphs,what='reco',suffix=''):
 '''plot graph representation of the event'''
 
 for iev,G in enumerate(graphs):
   print('===> In plotGraphs(), evt={}'.format(iev))

   for i in range(0,len(G)):
     #print(G.nodes[i]['features'][0])
     G.add_node(i, pos=list((G.nodes[i]['features'][1],G.nodes[i]['features'][2])))  # these represent the ieta,iphi coordinates!
   pos = nx.get_node_attributes(G, "pos")

   plt.figure()
   plt.ioff()
   nx.draw(G, pos, node_color="r", node_size=2, with_labels=False)

   plt.title('{w} hits, event {i}, {s}'.format(i=iev, w=what, s=suffix))
   plt.savefig('plots/graph{s}_{w}_event{i}.pdf'.format(s=suffix,i=iev, w=what))
   plt.savefig('plots/graph{s}_{w}_event{i}.png'.format(s=suffix,i=iev, w=what))
   plt.close()

def plotGraphKNN(iev,matrix,x,n_neighbours):
  '''converts a 2D np.array (matrix) into a networkx Graph'''
  print('===> In getGraphKNN(), evt={}'.format(iev))
  G=nx.from_numpy_array(matrix)
  for i in range(0,x.shape[0]):
    G.add_node(i, pos=list(x[i])) # networkx graphs prefer lists to numpy arrays...
  #G.nodes.data()
  pos = nx.get_node_attributes(G, "pos")
  plt.figure()
  plt.ioff()
  nx.draw(G, pos, node_color="r", node_size=2, with_labels=False)
  plt.title('reco KNN graph, k={k}, event={i}'.format(k=n_neighbours,i=iev))
  plt.savefig('plots/und_graphKNN{k}_reco_event{i}.pdf'.format(k=n_neighbours,i=iev))
  plt.savefig('plots/und_graphKNN{k}_reco_event{i}.png'.format(k=n_neighbours,i=iev))
  plt.close()

  #return G

def generateMatricesKNN(recodfs,n_neighbours):
  '''performs KNN search on each event and returns a list of adj-matrices (one per event)'''

  matrices = []

  for iev,df in enumerate(recodfs):
    x=np.column_stack((df.ieta,df.iphi))   
    print('iev={} shape x={}'.format(iev,x.shape))
   
    # now find the n nearest neighbours for each RH and use matrix representation of the result
    nbrs = NearestNeighbors(n_neighbors=n_neighbours, algorithm='kd_tree').fit(x)   # how to check euclidean distance ? or other distance, aka metric... 
    matrix = nbrs.kneighbors_graph(x).toarray()
    plotGraphKNN(iev,matrix,x,n_neighbours)
    matrices.append(matrix)

  return matrices

if __name__ == "__main__":
 
 nevts = 6 
 path = '/pnfs/psi.ch/cms/trivcat/store/user/mratti/Dumper/'
 #fns = [path + 'photon_E1to100GeV_closeEcal_EB_noPU_thrsLumi450_pfrh3.0_seed3.0_noMargin_thrRingEBXtalEE_shs1.0_maxd10.0_l450_P01_n10_njd0.root'] 
 #fns = [path + 'overlap_E1to100GeV_closeEcal_EB_noPU_thrsLumi450_pfrh3.0_seed3.0_noMargin_thrRingEBXtalEE_shs1.0_maxd10.0_l450_P02_n10_njd0.root']
 #fns = [path + 'overlap_E1to100GeV_closeEcal_EB_noPU_thrsLumi450_pfrh3.0_seed3.0_noMargin_thrRingEBXtalEE_shs1.0_maxd10.0_l450_P03_n10_njd0.root']
 fns = [path + 'overlap_E1to100GeV_closeEcal_EB_noPU_thrsLumi450_pfrh3.0_seed3.0_noMargin_thrRingEBXtalEE_shs1.0_maxd10.0_l450_P03_n30000_njd0.root']
 # use a list, as it's more general, can be extended to multiple files

 max_nevts = check(nevts=nevts, files=fns)
 
 # lists of dataframes, indexed by event number => for the purpose of plotting and conversion to graph
 #   dataframe format: 
 #      row = a given crystal
 #      column = a given feature
 simdfs_temp,recodfs = getDataFrames(nevts=max_nevts,files=fns)

 # prune the simdfs, keep only simhits with energy > 10 MeV
 simdfs = pruneDFs(dfs=simdfs_temp)
 
 # some x,y plotting   => SUPER SLOW, WHY?
 #plotXY(dfs=simdfs, what='sim')
 #plotXY(dfs=recodfs, what='reco')

 # KNN finding and adjacency matrix creation
 generateMatricesKNN(recodfs=recodfs, n_neighbours=15)
 matricesKNN = generateMatricesKNN(recodfs=recodfs, n_neighbours=10)
 generateMatricesKNN(recodfs=recodfs, n_neighbours=7)
 generateMatricesKNN(recodfs=recodfs, n_neighbours=3)

 # Create the dictionaries for reco and ground truth
 reco_dicts,true_dicts = getDictsFromDFs(recodfs,simdfs,doSelection=True,matricesSel=matricesKNN)

 # create the `graphs.GraphsTuple`  => these can be used as input to graph_nets library
 reco_graphs_np = gn.utils_np.data_dicts_to_graphs_tuple(reco_dicts) 
 true_graphs_np = gn.utils_np.data_dicts_to_graphs_tuple(true_dicts)

 # plot them using networkx
 plot_graphs_tuple_np(reco_graphs_np, label='reco')
 plot_graphs_tuple_np(true_graphs_np, label='true')
 #plot_compare_graphs([reco_graphs_np,true_graphs_np], labels=['reco', 'true'])
 #reco_graphs_np_noSel = gn.utils_np.data_dicts_to_graphs_tuple(reco_dicts_noSel) 
 # some graph plotting with networkx (not so informative at this point), but useful to test conversion
 #sim_graphs_nx =  gn.utils_np.graphs_tuple_to_networkxs(sim_graphs_tuple)
 reco_graphs_nx = gn.utils_np.graphs_tuple_to_networkxs(reco_graphs_np)
 #reco_graphs_nx_noSel = gn.utils_np.graphs_tuple_to_networkxs(reco_graphs_np_noSel
  
 #plotGraphs(graphs=reco_graphs_nx, suffix='KNN10')
 #plotGraphs(graphs=reco_graphs_nx_noSel, suffix='noSel')
 #plotGraphs(graphs=sim_graphs_nx)

