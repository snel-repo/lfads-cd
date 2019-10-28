import os
import h5py
import json

import numpy as np
import tensorflow as tf



def load_datasets(data_dir, data_filename_stem, hps):
  """Load the datasets from a specified directory.

  Example files look like
    >data_dir/my_dataset_first_day
    >data_dir/my_dataset_second_day

  If my_dataset (filename) stem is in the directory, the read routine will try
  and load it.  The datasets dictionary will then look like
  dataset['first_day'] -> (first day data dictionary)
  dataset['second_day'] -> (first day data dictionary)

  Args:
    data_dir: The directory from which to load the datasets.
    data_filename_stem: The stem of the filename for the datasets.

  Returns:
    datasets: a dataset dictionary, with one name->data dictionary pair for
    each dataset file.
  """
  print("Reading data from ", data_dir)
  datasets = read_datasets(data_dir, data_filename_stem)
  for k, data_dict in datasets.items():
    train_total_size = len(data_dict['train_data'])
    if train_total_size == 0:
      print("Did not load training set.")
    else:
      print("Found training set with number examples: ", train_total_size)
      if hps.cv_keep_ratio < 1.:
        np.random.seed(int(hps.cv_rand_seed))
        data_dict['train_data_cvmask'] = np.floor(hps.cv_keep_ratio +
                                          np.random.random_sample(data_dict['train_data'].shape)).astype(np.float32)
        np.random.seed()
    valid_total_size = len(data_dict['valid_data'])
    if valid_total_size == 0:
      print("Did not load validation set.")
    else:
      print("Found validation set with number examples: ", valid_total_size)
      if hps.cv_keep_ratio < 1.:
        np.random.seed(int(hps.cv_rand_seed))
        data_dict['valid_data_cvmask'] = np.floor(hps.cv_keep_ratio +
                                          np.random.random_sample(data_dict['valid_data'].shape)).astype(np.float32)
        np.random.seed()
    datasets[k] = clean_data_dict(data_dict)
  return datasets


def read_data(data_fname):
  """ Read saved data in HDF5 format.

  Args:
    data_fname: The filename of the file from which to read the data.
  Returns:
    A dictionary whose keys will vary depending on dataset (but should
    always contain the keys 'train_data' and 'valid_data') and whose
    values are numpy arrays.
  """

  try:
    with h5py.File(data_fname, 'r') as hf:
      data_dict = {k: np.array(v) for k, v in hf.items()}
      return data_dict
  except IOError:
    print("Cannot open %s for reading." % data_fname)
    raise



def read_datasets(data_path, data_fname_stem):
  """Read datasets in HDF5 format.

  This function assumes the dataset_dict is a mapping ( string ->
  to data_dict ).  It calls write_data for each data dictionary,
  post-fixing the data filename with the key of the dataset.

  Args:
    data_path: The path to the save directory.
    data_fname_stem: The filename stem of the file in which to write the data.
  """

  dataset_dict = {}
  fnames = os.listdir(data_path)

  print ('loading data from ' + data_path + ' with stem ' + data_fname_stem)
  for fname in fnames:
    if fname.startswith(data_fname_stem):
      data_dict = read_data(os.path.join(data_path,fname))
      idx = len(data_fname_stem) + 1
      key = fname[idx:]
      data_dict['data_dim'] = data_dict['train_data'].shape[2]
      data_dict['num_steps'] = data_dict['train_data'].shape[1]
      dataset_dict[key] = data_dict

  if len(dataset_dict) == 0:
    raise ValueError("Failed to load any datasets, are you sure that the "
                     "'--data_dir' and '--data_filename_stem' flag values "
                     "are correct?")

  print (str(len(dataset_dict)) + ' datasets loaded')
  return dataset_dict




def write_data(data_fname, data_dict, use_json=False, compression=None):
  """Write data in HDF5 format.

  Args:
    data_fname: The filename of teh file in which to write the data.
    data_dict:  The dictionary of data to write. The keys are strings
      and the values are numpy arrays.
    use_json (optional): human readable format for simple items
    compression (optional): The compression to use for h5py (disabled by
      default because the library borks on scalars, otherwise try 'gzip').
  """

  dir_name = os.path.dirname(data_fname)
  if not os.path.exists(dir_name):
    os.makedirs(dir_name)

  if use_json:
    the_file = open(data_fname,'w')
    json.dump(data_dict, the_file)
    the_file.close()
  else:
    try:
      with h5py.File(data_fname, 'w') as hf:
        for k, v in data_dict.items():
          clean_k = k.replace('/', '_')
          if clean_k is not k:
            print('Warning: saving variable with name: ', k, ' as ', clean_k)
          else:
            print('Saving variable with name: ', clean_k)
          hf.create_dataset(clean_k, data=v, compression=compression)
    except IOError:
      print("Cannot open %s for writing.", data_fname)
      raise


def clean_data_dict(data_dict):
  """Add some key/value pairs to the data dict, if they are missing.
  Args:
    data_dict - dictionary containing data for LFADS
  Returns:
    data_dict with some keys filled in, if they are absent.
  """

  keys = ['train_truth', 'train_ext_input', 'valid_data',
          'valid_truth', 'valid_ext_input', 'valid_train',
          'train_data_cvmask', 'valid_data_cvmask']
  for k in keys:
    if k not in data_dict:
      data_dict[k] = None

  return data_dict

