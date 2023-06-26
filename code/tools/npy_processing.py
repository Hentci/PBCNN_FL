import numpy as np
# web attack & DDOS-slowloris太小不用

# 加载.npy文件
data = np.load('/trainingData/sage/CIC-IDS2018-byte/CIC-IDS-2018/train/botnet.npy')

# 将数据转换为TensorFlow的Dataset对象
# dataset = tf.data.Dataset.from_tensor_slices(data)


print("Data shape:", data.shape)
print("Data content:")
print(data)