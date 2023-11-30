import os, sys, cv2, csv, argparse, random
import numpy as np
from keras.models import load_model
from tensorflow import keras
import tensorflow as tf
from tqdm import tqdm

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

def init_parameter():   
  parser = argparse.ArgumentParser(description='Test')
  parser.add_argument("--data", type=str, default='foo_test.csv', help="Dataset labels")
  parser.add_argument("--images", type=str, default='foo_test/', help="Dataset folder")
  parser.add_argument("--results", type=str, default='foo_results.csv', help="CSV file of the results")
  parser.add_argument("--model", type=str, default='ResNet152V2_gr01', help="Model folder to upload")
  args = parser.parse_args()
  return args

args = init_parameter()

print(">> LOADING MODEL")
cl = CustomLoss()

model_path = args.model
cs_obj = {"AAR_loss": cl.AAR_loss, "AAR_metric": cl.AAR_metric}
model = load_model(model_path, custom_objects = cs_obj)

# Print summary
model.summary()

# Reading CSV test file
with open(args.data, mode='r') as csv_file:
    gt = csv.reader(csv_file, delimiter=',')
    gt_num = 0
    gt_dict = {}
    for row in gt:
        gt_dict.update({row[0]: int(round(float(row[1])))})
        gt_num += 1
print(gt_num)

# Opening CSV results file
with open(args.results, 'w', newline='') as res_file:
    writer = csv.writer(res_file)
    # Processing all the images
    for image in tqdm(gt_dict.keys()):
        img = cv2.imread(args.images+ image)
        if img.size == 0:
            print("Error")
        img = tf.keras.applications.resnet_v2.preprocess_input(img)
        img = cv2.resize(img,(224,224))
        img = np.expand_dims(img, axis = 0)
        # Here you should add your code for applying your DCNN
        pred_age = int(np.round(model.predict(img, verbose=0)*100+1))
        # Writing a row in the CSV file
        writer.writerow([image, pred_age])

