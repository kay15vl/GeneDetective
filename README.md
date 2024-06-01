# GeneDetective
Term project for GNN (CSCI 5800) - Using the Seed Connector Algorithm and Graph Neural Network Methods for Disease Gene Classification Tasks



## **1. propagate_disgenet_genes.R**
**Input**: Disease gene association data form DisGeNET and MONO obo file.
**Output**: Propagated genes associations for the disease of interest (TB).

**Files needed**: 
1. mondo_2024_03.obo
2. all_gene_disease_associations.tsv

**Functions**
1. map_mondo_to_ontology (DisGeNET df, obo_file): Maps MONDO disease identifiers to the disease identifiers contained in DisGeNET database(UMLS).
2. get_ances (list of disease identifiers, obo file): gets the ancestor term of all the terms found in the DisGeNET data set.
3. propagate_genes (list of disease identifiers from DisGeNET df, ancester df, DisGeNET data with mapped MONDO identifiers): function that propagates genes from children terms up to parent terms.

**Output**: 
1. not_propagated_disgenet_genes.tsv
2. tb_propagated_disgenet_genes.tsv

## **2. get_negatives.ipynb
**Input**: BioGRID network data file and tb propagated genes collected from DisGeNET.
**Output**: List of negative genes used to train the logistic regression model

**Files needed**: 
1. mondo_2024_03.obo
2. all_gene_disease_associations.tsv

**Functions**
1. genes_in_netowork(positive gene list, adjacency matrix genes to index, adjacency matrix embeddings): function checks to see if the positive genes are in the network. If they are not, then they are taken out. Takes the rows corresponding to the positive list to create embeddings contining positive examples. A dictionary containing where the genes are indexed is outputted with the positive embeddings. 

2. get_negatives(positive gene list, djacency matrix genes to index, adjacency matrix embeddings): like the genes_in_network but instead embeddings for negatives is obtained. Only genes in the network are used as negative genes and are picked randomly. The same number of negatives are obtained as there are positives. A dictionary containing where the genes are indexed is outputted with the negative embeddings. 

**Output**: 
1. tb_negative_genes.tsv

## **3. final_gnn_code.ipynb**
**Input**: BioGRID data, propagated genes obtained from DisGeNET, and unpropagated genes obtained from DisGeNET,
**Output**: performance metric plots for each model (logistic regression and GNN models).

**Files needed**: 
1. not_propagated_disgenet_genes.tsv
2. tb_propagated_disgenet_genes.tsv
3. biogrid_network.txt
4. tb_negative_genes.tsv

**Functions**
1. genes_in_netowork(positive gene list, adjacency matrix genes to index, adjacency matrix embeddings): function checks to see if the positive genes are in the network. If they are not, then they are taken out. It takes the rows corresponding to the positive list to create embeddings that contain positive examples. A dictionary containing where the genes are indexed is outputted with the positive embeddings. 

2. load_negatives(adjacency matrix genes to index, adjacency matrix of the PPI, and file path to the file containing the negative examples): it reads in the negative genes form the `tb_negative_genes.tsv` file and gets the indexes of these genes within the adjacency matrix. The output is a list containing the negative genes and the index indicating where each gene is found withing the adjacency matrix.


3. get_feat_matrix (embeddings containing positive genes, embeddings containing negative genes, gene to index for the positive embeddings, gene to index for the negative embeddings): joins th enegative and positive embeddings to create the feature matrix for training. A dictionary containing where the genes are indexed is outputed with the feature matrix embeddings. 

4. get_target_vector (feature embeddings, gent to index dictionary of the feature embeddings): creates the target vector for the feature matrix.

5. get_embeddings (feature matrix): creates embeddings for the feature matrix.

6. class SeedConnector
	a. __init__(G, seed_set): initialization
	b. getLCC(subG): calculates the largest connected component (LCC) of the given subnetwork.
	c. get_SubG(nodes): creates a subgraph induced by the given set of nodes.
	d. getNeighbors(subG): finds all first-order neighbors of the nodes in the given subgraph.
	e. run_algo(): executes the Seed Connector Algorithm to expand the seed protein subnetwork.

7. build_module(seed_genes, G): builds module using the seed list.

8. label_module(seed, module,G): adding labels to the modules.

9. node_mapping(disgenet_df, index_col): creates a mapping from unique node identifiers to consecutive integers.

10. edge_list(disgenet_df, source_col, source_mapping, dst_col, dst_mapping): generates an edge list tensor using source and destination node mappings.

11. def get_data_with_features(disgenet_with)features, num_features): constructs the PyTorch Data object with features for diseases and genes.

12. def plot_roc_curve(title, model, data): plots ROC curve for GAE and VGAE models. 

13. plot_training_stats(title, losses, test_auc, test_ap, train_auc, train_ap): plots the AP and AUC after each epoch.

14. get_edge_dot_products( data, model, num_dz_nodes): computes the dot products between the encoded disease and gene nodes for the VGAE and GAE models.

15. get_ranked_edges(data_object, model, num_dz_ndoes): ranks the edges based on the dot products of the encoded nodes and returns the ranked edge list and dot products.

16. Class GCNEncoder(torch.nn.Module): defines and initializes the GAE model.

17. gae_train(train_data, gae_model, optimizer): trains the GAE model on the provided training data.

18. gae_test(test_data, gae_model): evaluates the GAE model on the provided test data.

19. Clas VariationalGCNEncoder(nn.Module): defines and initializes the VGAE model.

20. vgae_train(train_data, gae_model, optimizer): trains the VGAE model on the provided training data.

21. vgae_test(test_data, gae_model): evaluates the VGAE model on the provided test data.

## **4. figures folder**: contains png figures displaying the different performance metrics based on different parameters. Results are for both the logistic regression model and GNN models. This was part of the hyperparameter tuning. 

## **5. results folder**: contains the final performance metric results for the logistic regression and GNN models. These results are based on the optimal parameters for each model. 
