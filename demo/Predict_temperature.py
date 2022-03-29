import torch
import torch.optim as optim

# 数据
t_c = [0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0]   #预测值
t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]  #真实值
t_c = torch.tensor(t_c)
t_u = torch.tensor(t_u)

def model(t_u, w, b):
    return w * t_u + b

def loss_fn(t_p, t_c):
    squared_diffs = (t_p - t_c)**2
    return squared_diffs.mean()

def dloss_fn(t_p, t_c):
    dsq_diffs = 2 * (t_p - t_c) /t_p.size(0)
    return dsq_diffs

def dmodel_dw(t_u, w, b):
    return t_u

def dmodel_db(t_u, w, b):
    return 1.0

def grad_fn(t_u, t_c, t_p, w, b):
    dloss_dtp = dloss_fn(t_p, t_c)
    dloss_dw = dloss_dtp * dmodel_dw(t_u, w, b)
    dloss_db = dloss_dtp * dmodel_db(t_u, w, b)
    return torch.stack([dloss_dw.sum(), dloss_db.sum()])

def training_loop(n_epochs, learning_rate, params, t_u, t_c):
    for epoch in range(1, n_epochs + 1):
        w, b = params
        t_p = model(t_u, w, b)
        loss = loss_fn(t_p, t_c)
        grad = grad_fn(t_u, t_c, t_p, w, b)
        params = params - learning_rate * grad
        print('epoch %d, loss %f' % (epoch, float(loss)))

    return params

#pytorch计算图的自动求导机制
def auto_training_loop(n_epochs, learning_rate, params, t_u, t_c):
    for epoch in range(1, n_epochs + 1):
        if params.grad is not None:
            params.grad.zero_()

        t_p = model(t_u, *params)
        loss = loss_fn(t_p, t_c)
        loss.backward()

        with torch.no_grad():
            params -= learning_rate * params.grad

        if epoch % 500 == 0:
            print('epoch %d, loss %f' % (epoch, float(loss)))
    return params

#pytroch中的优化器
#每个优化器构造函数都接收一个参数列表(又称pytorch张量，通常将requires_grad设置为True)作为第一个输入，优化器可更新他们的值并访问他们的grad属性
def auto_optim_training_loop(n_epochs, optimizer, params, t_u, t_c):
    for epoch in range(1, n_epochs + 1):
        t_p = model(t_u, *params)
        loss = loss_fn(t_p, t_c)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            print('epoch %d, loss %f' % (epoch, float(loss)))
    return params



w = torch.ones(())
b = torch.zeros(())

t_p = model(t_u, w, b)
print("t_p:", t_p)
loss = loss_fn(t_p, t_c)
print('loss:', loss)

n_epochs=5000
learning_rate= 1e-5
params=torch.tensor([1.0, 0.0], requires_grad=True)
optimizer = optim.SGD([params], lr = learning_rate)

# auto_training_loop(n_epochs=5000, learning_rate= 1e-5, params=torch.tensor([1.0, 0.0], requires_grad=True), t_u = t_u, t_c = t_c)
auto_optim_training_loop(n_epochs = 5000, optimizer = optimizer, params = params, t_u = t_u, t_c = t_c)


