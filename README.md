# Musical onset detection using a CNN

Pytorch implementation of the method described in: </br>
Schlüter, Jan, and Sebastian Böck. "Improved musical onset detection with convolutional neural networks." 2014 ieee international conference on acoustics, speech and signal processing (icassp). IEEE, 2014.


## Requirements
* Pytorch
* Librosa
* Numpy
* Matplotlib(optional)

## Dataset (used in the paper)
* Can be obtained from here (until the Google drive links are alive) - <a href>"https://github.com/CPJKU/onset_db/issues/1#issuecomment-472300295"</a>

## Usage

### Train the network
1. Run <code>gen_songlist.py</code> to get the list of all songs for which there is onset annotation data available(there are some extra audios in the dataset)
2. Run <code>get_data_stats.py</code> to compute the mean and standard deviation across 80 mel bands over the entire dataset
3. Run <code>gen_data.py</code> to generate the 15-frame mel spectrogram chunks and frame-wise labels for all the audios
4. Run <code>train.py</code> to train the network. Specify a fold number in the command line when running this script. This is used to partition the data into train and val splits using the splits data provided by the authors. The training almost exactly follows the procedure described in the paper. The weights at the end of 100 epochs get saved in the models folder.

### Evaluate the network
1. Run <code>test.py</code> to evaluate on the dataset. Again, specify a fold number to get the results for that fold. Results get saved to a text file in the form of #true-postives, #false-alarms, and #ground-truth-onsets, summed over all the validation songs, for different evaluation thresholds.

### Load saved model
If you wish to use the trained model on different data, <code>utils.py</code> contains the model class definition (and some other helper functions). Import the model class from here and load one of the saved model state dicts from the models folder.
