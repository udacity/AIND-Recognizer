# Artificial Intelligence Engineer Nanodegree
## Probabilistic Models
## Project: Sign Language Recognition System

### Install

This project requires **Python 3** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [SciPy](https://www.scipy.org/)
- [scikit-learn](http://scikit-learn.org/0.17/install.html)
- [pandas](http://pandas.pydata.org/)
- [matplotlib](http://matplotlib.org/)
- [jupyter](http://ipython.org/notebook.html)
- [hmmlearn](http://hmmlearn.readthedocs.io/en/latest/)

Notes: 
1. It is highly recommended that you install the [Anaconda](http://continuum.io/downloads) distribution of Python and load the environment included in the "Your conda env for AI ND" lesson.
2. The most recent development version of hmmlearn, 0.2.1, contains a bugfix related to the log function, which is used in this project.  In order to install this version of hmmearn, install it directly from its repo with the following command from within your activated Anaconda environment:
```sh
pip install git+https://github.com/hmmlearn/hmmlearn.git
```

### Code

A template notebook is provided as `asl_recognizer.ipynb`. The notebook is a combination tutorial and submission document.  Some of the codebase and some of your implementation will be external to the notebook. For submission, complete the **Submission** sections of each part.  This will include running your implementations in code notebook cells, answering analysis questions, and passing provided unit tests provided in the codebase and called out in the notebook. 

### Run

In a terminal or command window, navigate to the top-level project directory `AIND_recognizer/` (that contains this README) and run one of the following command:

`jupyter notebook asl_recognizer.ipynb`

This will open the Jupyter Notebook software and notebook in your browser. Follow the instructions in the notebook for completing the project.


### Additional Information
##### Provided Raw Data

The data in the `asl_recognizer/data/` directory was derived from 
the [RWTH-BOSTON-104 Database](http://www-i6.informatik.rwth-aachen.de/~dreuw/database-rwth-boston-104.php). 
The handpositions (`hand_condensed.csv`) are pulled directly from 
the database [boston104.handpositions.rybach-forster-dreuw-2009-09-25.full.xml](boston104.handpositions.rybach-forster-dreuw-2009-09-25.full.xml). The three markers are:

*   0  speaker's left hand
*   1  speaker's right hand
*   2  speaker's nose
*   X and Y values of the video frame increase left to right and top to bottom.

Take a look at the sample [ASL recognizer video](http://www-i6.informatik.rwth-aachen.de/~dreuw/download/021.avi)
to see how the hand locations are tracked.

The videos are sentences with translations provided in the database.  
For purposes of this project, the sentences have been pre-segmented into words 
based on slow motion examination of the files.  
These segments are provided in the `train_words.csv` and `test_words.csv` files
in the form of start and end frames (inclusive).

The videos in the corpus include recordings from three different ASL speakers.
The mappings for the three speakers to video are included in the `speaker.csv` 
file.
