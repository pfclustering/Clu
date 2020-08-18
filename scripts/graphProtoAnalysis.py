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
                           'icP': events.array('simHit_icP')[inev], 
                         })  
    simdfs.append(simdf)

    recodf = pd.DataFrame({'energy': events.array('pfRecHit_energy')[inev],
                           'ieta': events.array('pfRecHit_ieta')[inev],
                           'iphi': events.array('pfRecHit_iphi')[inev],
                           })
    recodfs.append(recodf)

  return simdfs,recodfs

def createGraphsFromDFs(dfs, what='reco', doSelection=False, matricesSel=[]):
  '''converts a list of dataframes into a graphs_tuple compatible with graph_nets

     nodes_0 = [[10.0, 2, 3],  # Node 0
                [11.3, 0, 2],  # Node 1
                [34.3, 2, 1],  # Node 2
                [13.0, 3, 3],  # Node 3
                [14.0, 4, 4]]  # Node 4 
     
      edges_0 = [[100., 200.],  # Edge 0
              [101., 201.],  # Edge 1
              [102., 202.],  # Edge 2
              [103., 203.],  # Edge 3
              [104., 204.],  # Edge 4
              [105., 205.]]  # Edge 5

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

  data_dict_list = []
  for inev,df in enumerate(dfs):

    #define nodes, edges and globals
    nodes = []
    edges = []
    senders = []
    receivers = []
    globals = []

    # straight conversion from pd to list of lists
    nodes = df.values.tolist() # no!! if it has also icalo...
    
    # create edges between every node with every other node, in both directions
    for i,inode in enumerate(nodes):
      for j,jnode in enumerate(nodes):

        if doSelection: 
          #print('debug matrix =', matricesSel[inev][(i,j)])
          if matricesSel[inev][(i,j)]==0: continue
   
        if i==j: continue # do not create a self-edge
        #score = -1
        denergy = inode[0] - jnode[0]
        dieta =   inode[1] - jnode[1]
        diphi =   inode[2] - jnode[2]
        if what=='reco' or (what=='sim' and df.at[i,'icalo']==df.at[j,'icalo']) : 
          # if you're 'sim' only create edge if  inode and jnode belong to the same caloparticle
          edges.append([denergy,dieta,diphi]) # OK as features of the edges ???
          senders.append(i)
          receivers.append(j) 

    data_dict = {
      'globals': globals,
      'nodes': nodes,
      'edges': edges,
      'senders': senders,
      'receivers': receivers,
    }

    data_dict_list.append(data_dict)

  graphs_tuple = gn.utils_np.data_dicts_to_graphs_tuple(data_dict_list) 
  return graphs_tuple

def pruneDFs(dfs):
  '''return a list of dataframew where each is pruned with some specified selection criteria'''
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
  plt.savefig('plots/graphKNN{k}_reco_event{i}.pdf'.format(k=n_neighbours,i=iev))
  plt.savefig('plots/graphKNN{k}_reco_event{i}.png'.format(k=n_neighbours,i=iev))
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
 
 nevts = 8 
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
 plotXY(dfs=simdfs, what='sim')
 plotXY(dfs=recodfs, what='reco')


 # KNN analysis
 generateMatricesKNN(recodfs=recodfs, n_neighbours=15)
 matricesKNN = generateMatricesKNN(recodfs=recodfs, n_neighbours=10)
 generateMatricesKNN(recodfs=recodfs, n_neighbours=7)
 generateMatricesKNN(recodfs=recodfs, n_neighbours=3)
 # for each event, produce several plots showing the connections between points , after applying KNN 
 # return a list of networkx graphs, one per event 


 # This part is quite slow...
 # create the GN tuples  => these can be used as input to graph_nets library
 #sim_graphs_tuple =  createGraphsFromDFs(dfs=simdfs, what='sim')
 reco_graphs_tuple = createGraphsFromDFs(dfs=recodfs, what='reco', doSelection=True, matricesSel=matricesKNN)
 reco_graphs_tuple_noSel = createGraphsFromDFs(dfs=recodfs, what='reco', doSelection=False)
 # some graph plotting with networkx (not so informative at this point), but useful to test conversion
 #sim_graphs_nx =  gn.utils_np.graphs_tuple_to_networkxs(sim_graphs_tuple)
 reco_graphs_nx = gn.utils_np.graphs_tuple_to_networkxs(reco_graphs_tuple)
 reco_graphs_nx_noSel = gn.utils_np.graphs_tuple_to_networkxs(reco_graphs_tuple_noSel)
  
 plotGraphs(graphs=reco_graphs_nx, suffix='KNN10')
 plotGraphs(graphs=reco_graphs_nx_noSel, suffix='noSel')
 #plotGraphs(graphs=sim_graphs_nx)

