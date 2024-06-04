# gnn_qm9_models


## About the QM9 dataset
The QM9 dataset is a widely-used benchmark in the field of quantum chemistry and molecular machine learning. It contains a comprehensive collection of 134,000 small organic molecules composed of carbon (C), hydrogen (H), oxygen (O), nitrogen (N), and fluorine (F) atoms. Each molecule in the QM9 dataset is characterised by a set of features and multiple quantum mechanical properties, making it an ideal dataset for tasks such as molecular property prediction, quantum chemistry studies, and the development of graph neural networks.

Key features of the QM9 dataset:

- Molecular graphs: each molecule is represented as a graph where atoms correspond to nodes and bonds correspond to edges. This graph representation is fundamental for leveraging graph neural networks (GNNs) in property prediction tasks.

- Atomic features: the dataset includes 11 atomic features per node, such as atom type, valence, and hybridisation state, which provide rich information about the molecular structure.

- Quantum mechanical properties: the dataset includes 19 regression targets, representing various quantum mechanical properties such as the energy of the highest occupied molecular orbital (HOMO), the energy of the lowest unoccupied molecular orbital (LUMO), and the isotropic polarisability.

- Standardisation and availability: the dataset is standardised and publicly available, making it a reliable resource for benchmarking and comparing machine learning models in molecular studies.

## QM9 dataset preprocessing and statistics

### Dataset loading
The QM9 dataset, a standard dataset in molecule property prediction tasks, is loaded using the torch_geometric library. This dataset contains 130,831 molecular graphs, each represented with 11 features and 19 regression targets.

### Data shuffling and splitting
To ensure reproducibility and prevent any bias, the dataset is shuffled using a fixed random seed (42). The shuffled dataset is then split into training, validation, and test sets:

- Training set: 110,831 graphs

- Validation set: 10,000 graphs

- Test set: 10,000 graphs

This split ratio ensures a substantial amount of data is available for training while keeping enough samples for validation and testing to evaluate the model's performance.

### Normalisation
Normalisation is a crucial preprocessing step in machine learning, particularly for regression tasks. The target variable (property) at index 15 is normalised using the mean and standard deviation computed from the training set. This normalisation ensures that the model training is stable and the gradient descent converges more efficiently.

### Summary statistics
The dataset is summarised as follows:

- Number of graphs: 130,831

- Number of features: 11

- Number of regression Targets: 19

These statistics are printed to provide an overview of the dataset characteristics.

### Implementation summary
The preprocessing steps ensure the dataset is well-prepared for training Graph Neural Network (GNNs).

### Graph Convolutional Network (GraphClassificationModel)

#### Model architecture

The GraphClassificationModel model leverages the standard GCN layers to capture the structural and feature information of the molecules. The architecture comprises three GCN layers, followed by two fully connected (linear) layers.

The code implementation of the GCN model is modularised and made production-ready by defining separate classes for the GCN layer and the model itself.

#### Why 3 layers?

The choice of using three convolutional layers in the GCN model is based on the need to capture the hierarchical structure of the data. Each layer allows the model to learn increasingly complex representations. The first layer captures basic relationships, the second layer captures higher-order interactions, and the third layer further refines these representations. This depth is often sufficient to model complex datasets without introducing excessive computational complexity or overfitting.

#### Why 128 hidden channels?

The hidden channel size of 128 strikes a balance between model capacity and computational efficiency. It is large enough to capture intricate patterns in the data while being computationally manageable. Increasing the number of hidden channels can improve model performance, but it also increases the risk of overfitting and requires more computational resources.

#### Why 64 units in the fully connected layer?

The fully connected layer with 64 units is chosen to reduce the dimensionality of the learned representations before the final output layer. This layer helps in distilling the most important features from the convolutional layers, improving the model's ability to generalise. The choice of 64 units is a trade-off between preserving enough information and reducing complexity.

### GraphClassificationModel

The GraphClassificationModel was chosen for its efficiency and simplicity in leveraging convolutional operations on graphs. The three-layer structure allows for sufficient depth to capture the complex relationships within the molecular graphs without overfitting.

## Training the GCN model

### Architecture breakdown:

1. **Input features**:
   - **Number of features**: The input features for each node in the graph have a dimension of 11. This is specified when the `GraphClassificationModel` class is instantiated with `num_features=11`.

2. **Convolutional layers**:
   - **conv1 (GCNConv(11, 128))**:
     - **Input**: Node features of dimension 11.
     - **Output**: Node features of dimension 128.
     - **Purpose**: This layer aggregates and transforms the input features based on the graph structure, capturing the local structure and features of neighboring nodes.
   - **conv2 (GCNConv(128, 128))**:
     - **Input**: Node features of dimension 128 from `conv1`.
     - **Output**: Node features of dimension 128.
     - **Purpose**: This layer further refines the node features by considering the information from a wider neighborhood, enhancing the node representations.
   - **conv3 (GCNConv(128, 128))**:
     - **Input**: Node features of dimension 128 from `conv2`.
     - **Output**: Node features of dimension 128.
     - **Purpose**: This layer continues to aggregate and refine the node features, enabling more complex representations.

3. **Readout layer**:
   - **global_mean_pool**:
     - **Input**: Node features of dimension 128 from `conv3`.
     - **Output**: A single vector of dimension 128 for each graph (batch).
     - **Purpose**: This layer aggregates the node features into a graph-level representation by taking the mean of all node features in the graph. It effectively transforms node-level information to graph-level information.

4. **Fully connected (linear) layers**:
   - **lin1 (Linear(in_features=128, out_features=64))**:
     - **Input**: Pooled graph features of dimension 128.
     - **Output**: A 64-dimensional vector.
     - **Purpose**: This linear layer reduces the dimensionality of the graph-level representation and learns important features.
   - **lin2 (Linear(in_features=64, out_features=1))**:
     - **Input**: Features of dimension 64 from `lin1`.
     - **Output**: A single value (typically for regression tasks).
     - **Purpose**: This layer produces the final output, which is a single value per graph.

5. **Activation functions**:
   - **ReLU activation**: After each convolutional layer and the first linear layer, a ReLU (Rectified Linear Unit) activation function is applied. This introduces non-linearity into the model, allowing it to learn more complex patterns.

6. **Dropout**:
   - **Dropout (F.dropout)**:
     - **Input**: Features from `lin1`.
     - **Output**: A subset of the features with some elements randomly set to zero.
     - **Purpose**: Dropout is used to prevent overfitting by randomly dropping a fraction of the neurons during training. This encourages the model to learn more robust features.

### Forward Pass:
1. **Node embeddings**:
   - The input node features are passed through three GCNConv layers, each followed by a ReLU activation function. This process transforms the node features, capturing local graph structure and features of neighboring nodes.

2. **Readout layer**:
   - The node features are aggregated into a single graph-level representation using global mean pooling.

3. **Linear layers**:
   - The graph-level representation is passed through a fully connected layer (lin1) with ReLU activation, followed by dropout. Finally, the output is passed through another fully connected layer (lin2) to produce the final prediction.

### Purpose of each component:
- **GCNConv Layers**: Capture the structure of the graph and propagate information between neighboring nodes.
- **ReLU Activation**: Introduce non-linearity to the model, allowing it to learn more complex patterns.
- **Dropout**: Prevent overfitting by randomly dropping neurons during training.
- **Linear Layers**: Transform and reduce the dimensionality of the graph features to produce the final output.

### Overall flow:
1. **Input node features (11 dimensions)** are transformed and aggregated through three GCNConv layers, each producing 128-dimensional node features.
2. **The output of the last GCNConv layer is pooled using global mean pooling** to produce a fixed-size graph-level representation (128 dimensions).
3. **The pooled representation is passed through a linear layer** to reduce the dimensionality to 64.
4. **Finally, the 64-dimensional features are passed through another linear layer** to produce the final output (1 dimension).

This architecture is designed for a regression task on graph data, where the goal is to predict a continuous value for each graph. The GCN layers capture the local graph structure and propagate node features, while the linear layers transform these features to produce the final prediction.

# Insights & results

## Primary results of the GraphClassificationModel model

The primary evaluation metrics for the GCN model are as follows:

- Test Loss: 0.559448

- Test MAE: 0.738557

For the GraphClassificationModel model, the primary evaluation metric is the Mean Absolute Error (MAE) on the test set. The obtained MAE is 17.031112. This low MAE indicates that the GraphClassificationModel model is able to predict the target molecular properties with a relatively high degree of accuracy.

## Insights and explanations for the GraphClassificationModel model

The GCN model demonstrated a test loss of 0.559448 and a mean absolute error (MAE) of 0.738557. These results indicate that the GCN model performs well on the QM9 dataset for predicting molecular properties.


## Several factors contribute to the effectiveness of the GraphClassificationModel model

### Hierarchical learning

The GCN architecture's use of multiple layers enables the model to learn hierarchical features within the graph structure. Each layer in the GCN aggregates information from the nodes' local neighborhoods, progressively capturing higher-order relationships. This hierarchical feature extraction is crucial for understanding the complex molecular structures present in the QM9 dataset.

### Effective aggregation with global mean pooling

The global mean pooling operation aggregates node features into a graph-level representation, ensuring that the model captures the overall structural information of the molecule. By averaging the node features, global mean pooling provides a summary statistic that is invariant to the number of nodes in the graph, making it particularly suitable for variable-sized graphs like those in the QM9 dataset.

### Non-linearity with ReLU activation

The use of the ReLU activation function introduces non-linearity into the model, allowing it to capture more complex patterns in the data. ReLU helps to avoid the vanishing gradient problem and accelerates the convergence of the training process. This non-linearity is essential for the model to learn intricate molecular interactions.

### Regularisation with dropout

Incorporating dropout layers in the GCN model helps to prevent overfitting by randomly setting a fraction of the input units to zero during training. Dropout acts as a form of regularisation, forcing the model to learn robust features that generalise well to unseen data. This is particularly important given the complexity of the QM9 dataset, which contains diverse molecular structures.

### Depth and capacity with 3 layers and 128 hidden channels}

The choice of using three convolutional layers is motivated by the need to balance depth and computational efficiency. Three layers are sufficient to capture complex relationships in the data without overfitting or introducing excessive computational overhead. The hidden channel size of 128 strikes a balance between model capacity and computational efficiency. It is large enough to capture intricate patterns in the data while being computationally manageable. This depth and capacity allow the model to learn detailed representations of molecular structures.

### Dimensionality reduction with fully connected layers

The fully connected layer with 64 units is chosen to reduce the dimensionality of the learned representations before the final output layer. This layer helps in distilling the most important features from the convolutional layers, improving the model's ability to generalise. The choice of 64 units is a trade-off between preserving enough information and reducing complexity.

## Summary

The GCN model's architecture and design choices have proven effective for the task of molecular property prediction on the QM9 dataset. The combination of hierarchical learning through multiple GCN layers, effective aggregation with global mean pooling, non-linearity with ReLU activation, regularisation with dropout, and appropriate depth and capacity with hidden channels and fully connected layers contribute to the model's strong performance. These insights demonstrate the importance of careful architectural design in graph neural networks for achieving high performance in real-world applications.

![Predicted energy.png](imgs%2Fpredicted.png)

The plot presents the results of the Graph Convolutional Network (GCN) model's performance on the QM9 dataset, specifically for predicting the energy of molecules. The evaluation metrics and visualisation provide insights into the model's accuracy and reliability.

## Ideal fit line
The red line represents the ideal fit where the predicted energy values perfectly match the actual energy values. This line serves as a benchmark to assess the performance of the model.

## Scatter plot
Each point on the scatter plot represents a molecule, with its actual energy on the x-axis and the predicted energy on the y-axis. The color of the points indicates the number of atoms in the molecule, ranging from 5 to 25.

## Evaluation metrics:

- R² Score: = 0.922 indicates a strong correlation between the actual and predicted energy values. This high R² score suggests that the model explains a significant portion of the variance in the target variable.

- MAE: Mean Absolute Error (MAE) of 17.031 kcal/mol indicates the average absolute difference between the predicted and actual energy values. A lower MAE value signifies better model accuracy.

- RMSE: Root Mean Squared Error (RMSE) of 61.414 kcal/mol provides a measure of the average magnitude of the errors. RMSE is more sensitive to outliers than MAE.

## Insights

- Model accuracy: the high R² score of 0.922 demonstrates that the GCN model is highly effective in capturing the underlying patterns in the data and making accurate predictions.

- Error distribution: the scatter plot shows that most points are closely aligned with the ideal fit line, indicating accurate predictions. However, there are some deviations, particularly for molecules with higher energy values, which contribute to the overall RMSE.

- Impact of atom count: the color gradient in the scatter plot suggests that the number of atoms in a molecule does not significantly affect the accuracy of the model's predictions. Points are evenly distributed across different colors, indicating consistent performance across molecules of varying sizes.

## Conclusion

The GCN model demonstrates strong performance in predicting the energy of molecules in the QM9 dataset. The high R² score and low MAE values reflect the model's capability to capture the complex relationships within the molecular structures. Despite some deviations, the overall error distribution indicates reliable and accurate predictions. Future improvements could focus on reducing the outliers and further fine-tuning the model to enhance its predictive accuracy.

![Training and validation loss/Valudation MAE.png](imgs%2Fval.png)