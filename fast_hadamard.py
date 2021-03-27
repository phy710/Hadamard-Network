import numpy as np
import tensorflow as tf
from time import time
import matplotlib.pyplot as plt

num_features = 1024
input_size = 32
num = 10

def reverse(x, n):
    result = 0
    for i in range(n):
        if (x >> i) & 1: result |= 1 << (n - 1 - i)
    return result

def grayCode(n): 
    # Right Shift the number 
    # by 1 taking xor with  
    # original number 
    return n ^ (n >> 1) 

def fwht(a):
    num_features = a.shape.as_list()[-1]
    input_size = a.shape.as_list()[-2]
    #either 2 or 4 or 32 or 1024(naive matrix multiplication)
    radix_size = 32
    lg_n = np.log2(num_features).astype(int)//np.log2(radix_size).astype(int)
    X = tf.keras.layers.Reshape([input_size, input_size]+[radix_size]*lg_n)(a)
    transpose_dims = [0,1,2] + [lg_n-i-1+3 for i in range(lg_n)] 
    #print(transpose_dims)
    #Y = tf.transpose(X, transpose_dims)
    Y = X
    H = np.array([[1.,1],[1,-1]], dtype = np.float32)
    A_np = H
    for idx in range(np.log2(radix_size).astype(int)-1):
        A_np = np.kron(A_np, H)
    A_np_reorder = np.zeros((radix_size, radix_size), dtype = np.float32)
    for i in range(radix_size):
        A_np_reorder[grayCode(reverse(i, np.log2(radix_size).astype(int))), :] = A_np[i, :] 
    A = tf.constant(A_np_reorder)
    for idx in range(lg_n):
        Y = tf.tensordot(Y,A,(3+idx,1))
    
    # outputs = tf.reshape(Y, (input_size, num_features))
    Y = tf.transpose(Y, transpose_dims)
    #print(Y.get_shape())
    Y = tf.keras.layers.Reshape((input_size, input_size, num_features))(Y)
    return Y/np.sqrt(num_features)


f_1 = tf.keras.Input(shape=(input_size, input_size, num_features), name="input")
f_2 = fwht(f_1)


a = tf.math.abs(f_2)
b = tf.constant(np.arange(num_features)/(10*num_features), dtype=tf.float32)
r = tf.keras.layers.ReLU()(a-b)
t = tf.math.tanh(f_2)
f_3 = tf.keras.layers.Multiply()([t, r])

f_4 = fwht(f_3)
model = tf.keras.Model(inputs=f_1, outputs=f_4)
#model.compile()
model.summary()

x = np.random.rand(num, input_size, input_size, num_features).astype(np.float32)


start = time()
y = model.predict(x)
end = time()
time1 = end-start
  
saved_model_path = "./fwht/saved_model"
model.save(saved_model_path)

# TODO(b/156102192)
optimize_lite_model = False 

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)

lite_model_content = converter.convert()

with open("./fwht/lite_model.tflite", "wb") as f:
  f.write(lite_model_content)
print("Wrote %sTFLite model of %d bytes." %
      ("optimized " if optimize_lite_model else "", len(lite_model_content)))

# interpreter = tf.lite.Interpreter(model_content=lite_model_content)
interpreter = tf.lite.Interpreter(model_path="./fwht/lite_model.tflite")

yy = np.zeros((num, input_size, input_size, num_features)).astype(np.float32)
start = time()
for i in range(num):
    interpreter.allocate_tensors()
    interpreter.set_tensor(interpreter.get_input_details()[0]['index'], x[i:i+1, :, :])
    interpreter.invoke()
    yy[i:i+1, :, :] = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])
end = time()
time2 = end-start

print(time1)
print(time2)
print(sum(sum(sum(sum(abs(x-y)))))/(num*input_size*input_size*num_features))

fig = plt.figure()
plt.plot(range(100), x[0, 0, 0, :100])
plt.plot(range(100), x[1, 0, 0, :100])
plt.grid()
plt.title('x')
plt.show()

fig = plt.figure()
plt.plot(range(100), y[0, 0, 0, :100])
plt.plot(range(100), y[1, 0, 0, :100])
plt.grid()
plt.title('y')
plt.show()