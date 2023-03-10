<h1>Hybrid (Quantum-Classical) Autoencoder for Anomaly Detection.</h1> <br><br>

Please visit this repo (https://github.com/erenaykrcn/HAE_demonstrator) for the visual demonstrator of this project. Make sure to clone both repos in the same root directory. <br>

This project implements three different approaches for a Hybrid Autoencoder. 
The HAE module is based on the algorithm written by Alona Sakhnenko, Corey O'Meara, Kumar J. B. Ghosh, Christian B. Mendl, Giorgio Cortiana and Juan Bernabé-Moreno, which can be found here: <br>

https://arxiv.org/abs/2112.08869

<br>The HAE algorithm modifies the known approach of using a classical Autoencoder Neural Network to detect anomalies in a given data set to integrate a Quantum Circuit into the classical neural network to boost performance. The training of the weights are based on the recontruction loss, however evaluation of the given test data set is based on the assumption that the abnormal data points are outliers in the latent space representation of the high dimensional data. These outliers are detected through an IsolationForest model. This IsolationForest model is fitted (trained) by the latent space representation of the normal data. Then the test data is encoded to get the latent space representation and then fed into the IsolationForest to get the predictions.

<br>

![](https://github.com/erenaykrcn/HAE_demonstrator/blob/master/static/homepage/HAE_diagram.png)

<br>In addition to the aforementioned approach, I implemented and tested another approach for this problem, that is named QVC_loss approach in this project. Main difference of this approach is that the weights of the Quantum Circuit and weights of the Classical Autoencoder are trained using different loss functions/approaches. Firstly the classical weights are trained without the Quantum layer based on the reconstruction loss. Training data set for this step contains only normal data. Afterwards, a second training data set for the training of the hyperparameters of the Quantum layer is formed, containing both normal and abnormal data, which is also prelabeled. This data is first processed by the encoder of the Classical Autoencoder to get the latent space representation. This low dimensional data is then encoded to the encoding layer of the Quantum Circuit and hyperparameters of the processing layer of the Quantum Circuit are trained with a Variational Classifier Algorithm. SPSA Optimizer of Qiskit is used during training. The evaluation step of this approach does not require the use of the IsolationForest as the outcome of the Quantum layer already delivers the classification predictions.<br>


![](https://github.com/erenaykrcn/HAE_demonstrator/blob/master/static/homepage/QVC_diagram.png)

<br>

I observed a significant boost to the precision of the predictions, using the QVC_loss approach in comparison to the HAE (Hybrid Autoencoder) or CAE (Classical Autoencoder) approaches. Overall, QVC_loss also delivered higher F1 scores. the downside of the QVC_loss however is that it takes longer to be trained using a Quantum simulator.

