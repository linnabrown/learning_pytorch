

```python
import torch
torch.cuda.is_available()
```




    True




```python
Variable = torch.autograd.Variable
Tensor = torch.Tensor
```


```python
# Create Variable
x = Variable(Tensor([1]), requires_grad=True)
w = Variable(Tensor([2]), requires_grad=True)
b = Variable(Tensor([3]), requires_grad=True)

# Build Computational Graph

y = w * x + b

y.backward() #same as y.backward(torch.FloatTensor([1])) scala differentiate

print(x.grad)
print(w.grad)
print(b.grad)
```

    tensor([ 2.])
    tensor([ 1.])
    tensor([ 1.])



```python
x = torch.randn(3)
x = Variable(x, requires_grad = True)
x
```




    tensor([-1.0898, -0.1056, -0.8737])




```python
y = x * 2
print(y)
```

    tensor([-2.1796, -0.2113, -1.7473])



```python
y.backward(torch.FloatTensor([1, 0.1, 0.01]))
print(x.grad)
```

    tensor([ 2.0000,  0.2000,  0.0200])



```python
print(x.grad)
```

    tensor([ 2.0000,  0.2000,  0.0200])



```python
x
```




    tensor([-1.0898, -0.1056, -0.8737])




```python
y
```




    tensor([-2.1796, -0.2113, -1.7473])




```python
y.backward(torch.FloatTensor([1, 0.1, 0.01]))
print(x.grad)
```

    tensor([ 4.0000,  0.4000,  0.0400])



```python
 x

```




    tensor([-1.0898, -0.1056, -0.8737])




```python
y
```




    tensor([-2.1796, -0.2113, -1.7473])




```python
x.grad
```


```python

```
