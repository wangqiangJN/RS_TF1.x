'''
wq
tf1.15
'''
import tensorflow as tf
def target_attention(queries, keys, keys_length):
  '''
    queries:     [B, H]     当前item
    keys:        [B, T, H]  历史item
    keys_length: [B]        历史itemfeature长度
  '''
  queries_hidden_units = queries.get_shape().as_list()[-1]
  queries = tf.tile(queries, [1, tf.shape(keys)[1]])  
  queries = tf.reshape(queries, [-1, tf.shape(keys)[1], queries_hidden_units])
  din_all = tf.concat([queries, keys, queries-keys, queries*keys], axis=-1)
  d_layer_1_all = tf.compat.v1.layers.dense(din_all, 80, activation=tf.nn.relu, name='f1_att_dis_inf', reuse=tf.compat.v1.AUTO_REUSE)
  d_layer_2_all = tf.compat.v1.layers.dense(d_layer_1_all, 40, activation=tf.nn.relu, name='f2_att_dis_inf', reuse=tf.compat.v1.AUTO_REUSE)
  d_layer_3_all = tf.compat.v1.layers.dense(d_layer_2_all, 1, activation=None, name='f3_att_dis_inf', reuse=tf.compat.v1.AUTO_REUSE)
  d_layer_3_all = tf.reshape(d_layer_3_all, [-1, 1, tf.shape(keys)[1]])
  outputs = d_layer_3_all 
  # Mask
  key_masks = tf.sequence_mask(keys_length, tf.shape(keys)[1])   # [B, T]
  key_masks = tf.expand_dims(key_masks, 1) # [B, 1, T]
  paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
  outputs = tf.where(key_masks, outputs, paddings)  # [B, 1, T]
  # Scale
  outputs = outputs / (keys.get_shape().as_list()[-1] ** 0.5)
  # Activation
  outputs = tf.nn.sigmoid(outputs)  # [B, 1, T]
  # Weighted sum
  outputs = tf.matmul(outputs, keys)  # [B, 1, H]

  return outputs