# CPPNs and NeRF

The toy problem of memorizing a 2D image using an MLP is a nice problem to understand before going into solving the problem NeRF tackles (memorising a 3D scene given multiple view points). The task is to train an MLP to take image coordinates as input and produce the RGB value at that point as output.

<img width=500 src=imgs/toy_prob.png>

What is observed is that directly feeding in the image coordinates does not produce great results. Instead, we augment the input vector by running it through a positional encoding function where the input vector `v=(x, y)` is run through `sin` functions of increasing frequency.

<img width=500 src=imgs/nerf_vis2.png>

## Why does Positional Encoding help?

Random Fourier Features lets networks learn high frequency functions in low dimensional domains. Positional Encoding is a special case of Random Fourier Features where `B` is a power of 2.

<img width=500 src=imgs/nerf_vis.png>

Under certain conditions, NNs are effectively Kernel Regression. ReLU MLPs correspond to a dot product kernel. The Random Fourier Features is nothing but a kernel trick.

## Model Architecture 

MODEL_L is defined using the following logic
```python
inp_shape = (L_FACT-1)**2
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units = inp_shape, activation = 'ReLU', input_shape = (inp_shape, )))
for i in range(0, int(math.ceil(math.log2(inp_shape)))):
    units = max(3, inp_shape // 2**i)
    model.add(tf.keras.layers.Dense(units = units, activation = 'ReLU'))

model.compile(optimizer = 'adam', loss=keras.metrics.mean_squared_error, metrics = ['mse'])
```

Following is the Architecure of MODEL_33

<img height=500 src=imgs/model_33.png>

## Results

| Model | Training | Params | Size |
|--------|----------|-------|-----|
| Original | <img src=imgs/original.png>       | N/A    | 126.80 KB  |
| 33     | <img src=imgs/model_33.gif>       | 11,115    | 43.41 KB  |
| 65     | <img src=imgs/model_65.gif>       | 44,075    |  172.16 KB |
| 129    | <img src=imgs/model_129.gif>       | 175,531    | 685.66 KB  |
| 257    | <img src=imgs/model_257.gif>       | 700,587    | 2.67 MB  |

# Refrences

1. NeRF Video - https://www.youtube.com/watch?v=nRyOzHpcr4Q&t=1706s
2. Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains - https://arxiv.org/abs/2006.10739
3. Neural Tangent Kernel: Convergence and Generalization in Neural Networks - https://arxiv.org/abs/1806.07572
4. NeRF - https://arxiv.org/abs/2003.08934