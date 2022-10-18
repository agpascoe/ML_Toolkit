This pretends to be a machine learning toolkit. With several used routines in ML experiments as:
*  def plot_classModelDecisionMap(model, X, y, granuality=100):
  """
  Plots a map that shows how the model has defined the respectives areas of classification in a 2d plane

  X, y must be np array of dimenssion (,2), and (,n), where n depends upon how many classifications are
  """

* def createQwerties(clusters=1, nPerClust=100, blur=0.5, centroids=[[0,0]],  draw=True):
  """
  This function builds a set of data in up to 4 qwerties or clusters, each one with nPerclust points.
  The centroids of the qwerties is given by a np array of point (elements). Blur is a parameter that close up or not the qwerties,
  playing as a noice to each point.

   for example, to create a set of 200 points in two qwerties, with some intersection:
  createQwerties( clusters=2,
                   nPerClust = 100,
                  blur = 1.5,
                  centroids = np.array([[3,3],[5,3]]),
                  draw = True)
  
  The draw parameter indicates if the functions displays a graph of the querties distribution ot not.
  This function returns two datasets of points, and labels.
  
  """

* def splitData(partitions, batch_size, data, labels, verbose=False):
  """
  This function splits data and labels that come in np.array or tensor, and return three sets of data in
  pytorch dataloaders with batches of batch_size. For the splitting, uses skilearn train_test_split function.

  partitions is an array of [%trainning, %valid], where %test set is the remaining to get 100%
  """

* multiClass.ipynb is a notebook with a NN model based on pytorch 1.10.x of Multiclass clasiffication using a qwerties generator. The set of analisys of this model is fully completed and can be reuser for other models.

* multiClass-TF.ipynn which is a a Tensorflow 2.x and Keras 2.x implementation of the same multiclass model before

Also, this toolkit can be compiled and incorpored as globa library, following the nexts instructions:
> python setup.py bdist_wheel
/* be sure that setup.py consider the tags updated (or not)
> pip install ./dist/xxx-py3-none-any.whl /* this was generated by python setup.py before
...

/* then in your code or notebook, use:
> from jagpascoe_ML_toolkit.createDataFunctions import createQwerties, splitData, plot_classModelDecisionMap



