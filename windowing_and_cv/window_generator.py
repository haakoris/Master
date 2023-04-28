import numpy as np
import pandas as pd
import tensorflow as tf
from tscv_sliding import TimeSeriesSplitSliding
from sklearn.preprocessing import StandardScaler

class WindowGenerator():
  def __init__(self, input_width, label_width, shift,
               train_df=None, val_df=None, test_df=None,
               label_columns=None, n_splits=5, train_splits=3, test_splits=1,
               scale=True, scaler=StandardScaler):
    
    # Store the raw data.
    self.train_df = train_df
    self.val_df = val_df
    self.test_df = test_df
    self.n_splits = n_splits
    self.train_splits = train_splits
    self.test_splits = test_splits
    self.scale = scale
    self.scaler = scaler

    # Work out the label column indices.
    self.label_columns = label_columns
    if label_columns is not None:
      self.label_columns_indices = {name: i for i, name in
                                    enumerate(label_columns)}
    self.column_indices = {name: i for i, name in
                           enumerate(train_df.columns)}

    # Work out the window parameters.
    self.input_width = input_width
    self.label_width = label_width
    self.shift = shift

    self.total_window_size = input_width + shift

    self.input_slice = slice(0, input_width)
    self.input_indices = np.arange(self.total_window_size)[self.input_slice]

    self.label_start = self.total_window_size - self.label_width
    self.labels_slice = slice(self.label_start, None)
    self.label_indices = np.arange(self.total_window_size)[self.labels_slice]


  def __repr__(self):
    return '\n'.join([
        f'Total window size: {self.total_window_size}',
        f'Input indices: {self.input_indices}',
        f'Label indices: {self.label_indices}',
        f'Label column name(s): {self.label_columns}'])
  
  def split_window(self, features):
    inputs = features[:, self.input_slice, :]
    labels = features[:, self.labels_slice, :]
    if self.label_columns is not None:
      labels = tf.stack(
          [labels[:, :, self.column_indices[name]] for name in self.label_columns],
          axis=-1)

    # Slicing doesn't preserve static shape information, so set the shapes
    # manually. This way the `tf.data.Datasets` are easier to inspect.
    inputs.set_shape([None, self.input_width, None])
    labels.set_shape([None, self.label_width, None])

    return inputs, labels

  def make_dataset(self, data):
    data = np.array(data, dtype=np.float32)
    ds = tf.keras.utils.timeseries_dataset_from_array(
        data=data,
        targets=None,
        sequence_length=self.total_window_size,
        sequence_stride=1,
        shuffle=True,
        batch_size=32,)

    ds = ds.map(self.split_window)

    return ds
  
  @property
  def train(self):
    return self.make_dataset(self.train_df)

  @property
  def val(self):
    return self.make_dataset(self.val_df)

  @property
  def test(self):
    return self.make_dataset(self.test_df)

  @property
  def folds(self, scale=True, scaler=StandardScaler):
    '''
    Create datasets for time series sliding window cross validation
    '''
    tscv = TimeSeriesSplitSliding(n_splits=self.n_splits, train_splits=self.train_splits,
                                  test_splits=self.test_splits, fixed_length=True)
    cv_indices = tscv.split(self.train_df)
    cv_folds = []
    for i in range(self.n_splits):
      train_indices, val_indices = next(cv_indices)
      train = self.train_df.iloc[train_indices]
      val = self.train_df.iloc[val_indices]
      if self.scale:
        scale = self.scaler()
        train = scale.fit_transform(train)
        val = scale.transform(val)
      cv_folds.append([self.make_dataset(train), self.make_dataset(val)])
    return np.array(cv_folds).reshape((self.n_splits, 2))

  @property
  def np_folds(self):
    '''
    Create numpy arrays for time series sliding window cross validation
    '''
    tscv = TimeSeriesSplitSliding(n_splits=self.n_splits, train_splits=self.train_splits,
                                  test_splits=self.test_splits, fixed_length=True)
    cv_indices = tscv.split(self.train_df)
    cv_folds = []
    for i in range(self.n_splits):
      train_indices, val_indices = next(cv_indices)
      train = self.train_df.iloc[train_indices]
      val = self.train_df.iloc[val_indices]     
      if self.scale:
        scale = self.scaler()
        train = scale.fit_transform(train)
        val = scale.transform(val)
      train, val = self.make_dataset(train), self.make_dataset(val)
      train_X = np.concatenate([x for x, y in train], axis=0)
      train_y = np.concatenate([y for x, y in train], axis=0)
      val_X = np.concatenate([x for x, y in val], axis=0)
      val_y = np.concatenate([y for x, y in val], axis=0)
      cv_folds.append([[train_X, train_y, val_X, val_y]])
    return np.array(cv_folds, dtype=object).reshape((self.n_splits, 4))

  @property
  def example(self):
    """Get and cache an example batch of `inputs, labels` for plotting."""
    result = getattr(self, '_example', None)
    if result is None:
      # No example batch was found, so get one from the `.train` dataset
      result = next(iter(self.train))
      # And cache it for next time
      self._example = result
    return result
  

if __name__ == '__main__':
  path = "/Users/eliassovikgunnarsson/Downloads/vix-daily_csv.csv"
  df = pd.read_csv(path, header = 0, index_col= 0)
  n = len(df)
  train_df = df[0:int(n*0.7)]
  val_df = df[int(n*0.7):int(n*0.9)]
  test_df = df[int(n*0.9):]

  w1 = WindowGenerator(30, 1, 1, train_df=train_df, val_df=val_df, test_df=test_df, label_columns=['VIX Close'])
  print(w1)

  folds = w1.np_folds