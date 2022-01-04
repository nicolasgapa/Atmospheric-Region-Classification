# Categorizing Characteristic Regions of Nightside High-Latitude Ionospheric Irregularities Using a Machine Learning Approach

Authors: Nicolas Gachancipa, Chintan Thakrar, Kshitija Deshpande

Global Navigation Satellite Systems (GNSS) team 

Space Physics Research Laboratory - Embry-Riddle Aeronautical University

More about the GNSS team at http://pages.erau.edu/~dbgnss/website_main.php

**Procedure:**
1. Download the python script and the training data.
2. Install the required dependencies. The script needs numpy, pandas, matplotlib, and tensorflow to run. 
3. Define inputs:
    1. split: Percentage of data that will be used for training. E.g., split = 0.8 will use 80% of the data for training and 20% for testing.
    2. file_name: csv file containing traing data. E.g., file_name = 'training_data.csv' for the file included in this repository. 
    3. model_type: Activation function used in the neural network. 'sigmoid' or 'softmax'. The softmax function is recommended since it is often used in classification models. The sigmoid function returns a value between 0 and 1, where 0 is polar cap and 1 is auroral oval. Conversely, the softmax fucntion returns two probailities, each corresponding to one of the regions. 
    4. epochs: Number of training epochs. 
    5. plot_loss: Plot the loss history (epochs vs loss). Boolean (True or False). 
