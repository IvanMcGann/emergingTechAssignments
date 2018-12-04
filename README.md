# emergingTechAssignments
Emerging Technologies Assignments


How to use Python - 

Install Anaconda:
1. Visit https://www.anaconda.com/download/
2. Download the latest version of anaconda and run the exe file.
3. Read through the boxes and select all applicable values.
4. Once downloaded it will give you access to many libraries and features including using Jupyter Notebook.
5. After its installed as a precaution make sure you update all packages using the following input in the command line - conda update --all
6. That may take some time to install all updates and requires an internet connection.
7. Not all libraries are installed by default so you may be required to install other libraries.
8. install the following plugins on Anaconda to run the python script.
	conda install scikit-learn
	conda install -c anaconda keras

Run Jupyter Notebook:
1. Once Anaconda has been successfully installed it gives you access to a web application called Jupyter Notebook.
2. To run this you enter the designated directory or folder and in the cmd/cmnder console type jupyter notebook and press enter, the application will then open in the default browser.
3. Being a web app it requires a secure internet connection.
4. Each notebook is a Python3 file.
5. Once you chose a file e.g. numpy-random.ipynb all the cells will run automatically, although some may take more time than other.
6. If you make any alterations you may run into errors that can be as a result of a library import or a prior value not being found, in this circumstances rerun the notebook.
7. Do this by clicking on the kernel button on the menu and select restart and run all.


Run Python Script: 
1.  In the folder that contains the script type into the command line - python digitrec.py 
2.  After a short time you will eventually be presented with the menu.
3.  If you have a model made press 1 and enter the name.
4.  If you press 2 the model will be created automatically.
5.  Next train the model by pressing 3 this will allow the model to learn the mnist training images and labels.
6.  Press 4 to test if the model is found and display its summary.
7.  Press 5 to save the model in h5 format e.g. testMod.h5
8.  Press 6 to test the model accuracy against png images on the system.
9.  You are then asked the name of the images, I saved the png's in an images folder. so I will enter that directory first.
10. Each png file is called numN.png - n meaning the number. so an example of a the command input would be images/num1.png
11. Each numbers have varying levels of accuracy.
12. You can use option 1 to use the previously saved model rather than creating a new one. All models withold the same parameters,
13. Press option 7 to exit and close the menu

Links Used:

Anaconda:

https://stackoverflow.com/questions/45197777/how-do-i-update-anaconda

https://anaconda.org/conda-forge/keras

https://scikit-learn.org/stable/install.html

https://www.tutorialspoint.com/index.htm


Markdown tips in notebook:

https://www.ibm.com/support/knowledgecenter/SSQNUZ_current/com.ibm.icpdata.doc/dsx/markd-jupyter.html

https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet

https://www.tutorialspoint.com/index.htm

https://jupyter-notebook.readthedocs.io/en/stable/examples/Notebook/Working%20With%20Markdown%20Cells.html

http://csrgxtu.github.io/2015/03/20/Writing-Mathematic-Fomulars-in-Markdown/

https://stackoverflow.com/questions/32370281/how-to-embed-image-or-picture-in-jupyter-notebook-either-from-a-local-machine-o


Notebook tips: 

https://en.wikipedia.org/

http://muthu.co/understanding-binomial-distribution-using-python/

https://bigdata-madesimple.com/how-to-implement-these-5-powerful-probability-distributions-in-python/

https://docs.scipy.org/doc/numpy-1.15.1/reference/routines.random.html

https://pynative.com/python-random-seed/

https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.iloc.html

https://www.kaggle.com/jchen2186/machine-learning-with-iris-dataset



Plot tips:

https://seaborn.pydata.org/generated/seaborn.scatterplot.html

https://seaborn.pydata.org/tutorial/distributions.html

https://seaborn.pydata.org/tutorial/aesthetics.html

https://matplotlib.org/api/_as_gen/matplotlib.pyplot

http://pandas.pydata.org/pandas-docs/version/0.13/visualization.html

https://seaborn.pydata.org/tutorial/color_palettes.html


MNIST - Digirec: 

http://yann.lecun.com/exdb/mnist/

https://towardsdatascience.com/a-simple-2d-cnn-for-mnist-digit-recognition-a998dbc1e79a

https://www.digitalocean.com/community/tutorials/how-to-build-a-neural-network-to-recognize-handwritten-digits-with-tensorflow

https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/

https://machinelearningmastery.com/handwritten-digit-recognition-using-convolutional-neural-networks-python-keras/

https://bigdata-madesimple.com/how-to-implement-these-5-powerful-probability-distributions-in-pytho

https://nextjournal.com/gkoehler/digit-recognition-with-keras

https://docs.python.org/2/library/gzip.html

https://github.com/ianmcloughlin/jupyter-teaching-notebooks

https://www.programiz.com/python-programming/global-keyword#rules-global

