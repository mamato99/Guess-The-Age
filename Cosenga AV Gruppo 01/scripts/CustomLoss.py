import tensorflow as tf

class CustomLoss():

  def __init__(self, alpha=0.5, beta=0.5, verbose=0):
    self.alpha = alpha
    self.beta = beta
    self.verbose = verbose

  @tf.function
  def _calculus(self, y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    mae_fun = tf.keras.losses.MeanAbsoluteError()
    mae = mae_fun(y_true, y_pred)

    condition = tf.less_equal(y_true, 0.1)
    indices = tf.where(condition)
    mae_1 = mae_fun(tf.gather(y_true, indices), tf.gather(y_pred, indices))
    mae_diff_1 = tf.square(mae_1-mae)

    condition1 = tf.less_equal(y_true, 0.2)
    condition2 = tf.greater(y_true, 0.1)
    condition = tf.logical_and(condition1, condition2)
    indices = tf.where(condition)
    mae_2 = mae_fun(tf.gather(y_true, indices), tf.gather(y_pred, indices))
    mae_diff_2 = tf.square(mae_2-mae)

    condition1 = tf.less_equal(y_true, 0.3)
    condition2 = tf.greater(y_true, 0.2)
    condition = tf.logical_and(condition1, condition2)
    indices = tf.where(condition)
    mae_3 = mae_fun(tf.gather(y_true, indices), tf.gather(y_pred, indices))
    mae_diff_3 = tf.square(mae_3-mae)

    condition1 = tf.less_equal(y_true, 0.4)
    condition2 = tf.greater(y_true, 0.3)
    condition = tf.logical_and(condition1, condition2)
    indices = tf.where(condition)
    mae_4 = mae_fun(tf.gather(y_true, indices), tf.gather(y_pred, indices))
    mae_diff_4 = tf.square(mae_4-mae)

    condition1 = tf.less_equal(y_true, 0.5)
    condition2 = tf.greater(y_true, 0.4)
    condition = tf.logical_and(condition1, condition2)
    indices = tf.where(condition)
    mae_5 = mae_fun(tf.gather(y_true, indices), tf.gather(y_pred, indices))
    mae_diff_5 = tf.square(mae_5-mae)

    condition1 = tf.less_equal(y_true, 0.6)
    condition2 = tf.greater(y_true, 0.5)
    condition = tf.logical_and(condition1, condition2)
    indices = tf.where(condition)
    mae_6 = mae_fun(tf.gather(y_true, indices), tf.gather(y_pred, indices))
    mae_diff_6 = tf.square(mae_6-mae)

    condition1 = tf.less_equal(y_true, 0.7)
    condition2 = tf.greater(y_true, 0.6)
    condition = tf.logical_and(condition1, condition2)
    indices = tf.where(condition)
    mae_7 = mae_fun(tf.gather(y_true, indices), tf.gather(y_pred, indices))
    mae_diff_7 = tf.square(mae_7-mae)

    condition = tf.greater(y_true, 0.7)
    indices = tf.where(condition)
    mae_8 = mae_fun(tf.gather(y_true, indices), tf.gather(y_pred, indices))
    mae_diff_8 = tf.square(mae_8-mae)

    mae_temp = tf.stack([mae_1, mae_2, mae_3, mae_4, mae_5, mae_6, mae_7, mae_8], axis=0)
    indices = tf.where(tf.logical_not(tf.math.is_nan(mae_temp)))
    mae_total = tf.gather(mae_temp, indices)
    mmae = tf.reduce_mean(mae_total)
    mae_diff_temp = tf.stack([mae_diff_1, mae_diff_2, mae_diff_3, mae_diff_4, mae_diff_5, mae_diff_6, mae_diff_7, mae_diff_8])
    variance_temp = tf.gather(mae_diff_temp, indices)
    variance = tf.sqrt(tf.reduce_mean(variance_temp))

    if self.verbose > 0:        
      tf.print('maej:',end=' ')
      for e in tf.transpose(mae_total)[0]:
        tf.print(e*100,end=' ')
      tf.print('\tmae:',mae*100,'\tmmae:',mmae*100,'\tstd:',variance*100)

    return mmae, variance

  @tf.function
  def AAR_loss(self, y_true, y_pred):
    mmae, variance = self._calculus(y_true, y_pred)
    return self.alpha*mmae + self.beta*variance

  @tf.function
  def AAR_metric(self, y_true, y_pred):
      
    mmae, variance = self._calculus(y_true, y_pred)

    #𝐴𝐴𝑅 = max(0; 5 − 𝑚𝑀𝐴𝐸) + max(0; 5 − 𝜎)
    AAR = tf.math.maximum(tf.zeros((1,)),tf.constant(5, dtype=tf.float32)-mmae*100)+tf.math.maximum(tf.zeros((1,)),tf.constant(5, dtype=tf.float32)-variance*100)
      
    if self.verbose > 0:
      tf.print('AAR:',AAR[0])
      
    return AAR
