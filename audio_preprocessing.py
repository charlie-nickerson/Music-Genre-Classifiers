import os      # FILE PROCESSING LIBRARY
import librosa # AUDIO PROCESSING LIBRARY
import math
import json
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning) # Removes Future Warnings

DATASET_PATH = "genres_original"
JSON_PATH = "data.json"

SAMPLE_RATE = 22050
DURATION = 30
SAMPLES_PER_TRACK = SAMPLE_RATE*DURATION

def save_MelFreqCo(dataset_path, json_path, n_mfcc=13, n_fft=2048, hop_length=512, n_segments=5):

  # Store data into dictionary
  data = {
      "mapping":[], # List that maps the different genre labels. Ex: ["classical", "rock"]
      "mfcc":[],    # The training inputs
      "labels":[]   # The targets [0,0,1]
  }
  n_samples_per_segment = int(SAMPLES_PER_TRACK) / n_segments
  expected_num_mfcc_vectors_per_seg = math.ceil(n_samples_per_segment / hop_length)
  # Loop through all the genres
  # dirpath is the name of the current folder your working in
  # dirname is a list of all the sub-folder in dirpath
  # filenames is a list of all the files in dirpath
  for i, (dirpath, dirname, filenames) in enumerate(os.walk(dataset_path)):
    # Check if dirname is at the root
    if dirpath is not dataset_path:
      dirpath_components = dirpath.split("/") # genres_origal/blues --> ["genre_original", "rock"]
      s_label = dirpath_components[-1]        # Saves the semantic label
      data["mapping"].append(s_label)         # Adds s_label to the end of mapping
      print("\nProcessing {}".format(s_label))
      
      for file in filenames:
        # Load in the audio file
        file_path = os.path.join(dirpath, file)
        signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

        #process segments. extract mfcc and store data
        for s in range(n_segments):
            start_sample = int(n_samples_per_segment * s)
            end_sample = int(start_sample + n_samples_per_segment)

            mfcc = librosa.feature.mfcc(signal[start_sample:end_sample], 
                                        sr=sr,
                                        n_fft=n_fft,
                                        hop_length=hop_length)
            mfcc = mfcc.T

            # Store the mfcc into segments if it meets the expected length
            if len(mfcc) == expected_num_mfcc_vectors_per_seg:
                data["mfcc"].append(mfcc.tolist()) # must convert to list in order to be stored in the json file
                data["labels"].append(i-1)
                print("{}, segment:{}".format(file_path, s))
    
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)

if __name__ == "__main__":
    save_MelFreqCo(DATASET_PATH, JSON_PATH, n_segments=10)