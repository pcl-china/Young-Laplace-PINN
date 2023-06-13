from collections import OrderedDict
import torch
import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.tri as tri
import shapely.geometry
import pandas as pd

# 定义网络
class DNN(torch.nn.Module):
    def __init__(self, layers):
        super(DNN, self).__init__()
        self.depth = len(layers) - 1
        self.activation = torch.nn.Tanh
        layer_list = list()
        for i in range(self.depth - 1):
            layer_list.append(('layer_%d' % i, torch.nn.Linear(layers[i], layers[i + 1])))
            layer_list.append(('activation_%d' % i, self.activation()))
        layer_list.append(
            ('layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1])))
        layerDict = OrderedDict(layer_list)
        self.layers = torch.nn.Sequential(layerDict).double()

    def forward(self, x):
        out = self.layers(x)
        return out

# CUDA support
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

class PhysicsInformedNN():
    # 初始化
    def __init__(self, x_area, x_bc, n_bc, cos_angle, dxdy, layers,Bo,ub,lb):
        self.x = torch.tensor(x_area[:, 0:1], requires_grad=True).double().to(device)
        self.y = torch.tensor(x_area[:, 1:2], requires_grad=True).double().to(device)
        self.x_bc = torch.tensor(x_bc[:, 0:1], requires_grad=True).double().to(device)
        self.y_bc = torch.tensor(x_bc[:, 1:2], requires_grad=True).double().to(device)
        self.n_bcx = torch.tensor(n_bc[:, 0:1]).double().to(device)
        self.n_bcy = torch.tensor(n_bc[:, 1:2]).double().to(device)
        self.cos_angle = torch.tensor(cos_angle).double().to(device)
        self.ub = torch.tensor(ub).double().to(device)
        self.lb = torch.tensor(lb).double().to(device)
        self.lambda_lamb = torch.tensor([0.0], requires_grad=True).to(device)
        self.lambda_lamb = torch.nn.Parameter(self.lambda_lamb)
        self.Bo = Bo
        self.layers = layers
        self.dxdy = dxdy
        # 定义一个深度网络
        self.dnn = DNN(layers).to(device)
        self.dnn.register_parameter('lambda_lamb', self.lambda_lamb)

        self.optimizer_LBFGS = torch.optim.LBFGS(
            self.dnn.parameters(),
            lr=1.0,
            max_iter=20000,
            max_eval=20000,
            history_size=50,
            tolerance_grad=1.0 * np.finfo(float).eps,
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe"  # can be "strong_wolfe"
        )
        self.optimizer_adam = torch.optim.Adam(self.dnn.parameters(), 0.001)
        self.iter = 0
        self.loss = []

    def net_u(self, x, y):
        X = torch.cat([x, y], dim=1)
        X = 2.0 * (X-self.lb)/(self.ub - self.lb) - 1.0
        u = self.dnn(X)
        return u

    def net_f(self, x, y):
        u = self.net_u(x, y)

        u_x = torch.autograd.grad(u, x,grad_outputs=torch.ones_like(u),retain_graph=True,create_graph=True)[0]
        u_y = torch.autograd.grad(u, y,grad_outputs=torch.ones_like(u),retain_graph=True,create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x,grad_outputs=torch.ones_like(u_x),retain_graph=True,create_graph=True)[0]
        u_xy = torch.autograd.grad(u_x, y,grad_outputs=torch.ones_like(u_x),retain_graph=True,create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, y,grad_outputs=torch.ones_like(u_y),retain_graph=True,create_graph=True)[0]

        f = (u_xx * (1 + u_y ** 2) + u_yy * (1 + u_x ** 2) - 2 * u_x * u_xy * u_y)/(1 + u_x ** 2 + u_y ** 2) ** 1.5\
            +(- u * self.Bo + self.lambda_lamb)
        v = sum(u) * self.dxdy
        return f,v

    def net_bc_natural(self, x, y):
        u = self.net_u(x, y)
        u_x = torch.autograd.grad(u, x,grad_outputs=torch.ones_like(u),retain_graph=True,create_graph=True)[0]
        u_y = torch.autograd.grad(u, y,grad_outputs=torch.ones_like(u),retain_graph=True,create_graph=True)[0]
        bc = (u_x*self.n_bcx+u_y*self.n_bcy)/(1 + u_x ** 2 + u_y ** 2) ** .5 - self.cos_angle
        return bc

    def Calculate_loss(self):
        bc = self.net_bc_natural(self.x_bc, self.y_bc)
        f, v = self.net_f(self.x, self.y)
        f_bc = self.net_f(self.x_bc, self.y_bc)[0]
        f = torch.cat((f, f_bc), dim=0)
        loss_bc = torch.mean(bc ** 2)
        loss_f = torch.mean(f ** 2)
        loss_v = (v*self.Bo - 2 * self.cos_angle * math.pi) ** 2
        loss = 100 * loss_bc + 100 * loss_f + 1 * loss_v

        self.iter += 1
        if self.iter % 1000 == 0:
            print(
                'Iter %d, Loss: %.5e, Loss_bc: %.5e, Loss_f: %.5e, Loss_v: %.5e' % (
                    self.iter, loss.item(), loss_bc.item(), loss_f.item(), loss_v.item())
            )
            print(
                'lambda: %.5e' % (
                    self.lambda_lamb)
            )
        self.loss.append([self.iter, loss.item(), loss_bc.item(), loss_f.item(), loss_v.item()])
        return loss

    def loss_func(self):
        self.optimizer_LBFGS.zero_grad()
        loss = self.Calculate_loss()
        loss.backward()
        return loss

    def train(self):
        self.dnn.train()
        # 先使用Adam预训练
        for i in range(1,1000):
            self.optimizer_adam.zero_grad()
            loss = self.Calculate_loss()
            loss.backward()
            self.optimizer_adam.step()
        # 再使用LBGFS
        self.optimizer_LBFGS.step(self.loss_func)


    def predict(self, X):
        x = torch.tensor(X[:, 0:1], requires_grad=True).double().to(device)
        y = torch.tensor(X[:, 1:2], requires_grad=True).double().to(device)
        self.dnn.eval()
        u = self.net_u(x, y)
        f_pde,v = self.net_f(x, y)
        u_x = torch.autograd.grad(u, x,grad_outputs=torch.ones_like(u),retain_graph=True,create_graph=True)[0]
        u_y = torch.autograd.grad(u, y,grad_outputs=torch.ones_like(u),retain_graph=True,create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x,grad_outputs=torch.ones_like(u_x),retain_graph=True,create_graph=True)[0]
        u_xy = torch.autograd.grad(u_x, y,grad_outputs=torch.ones_like(u_x),retain_graph=True,create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, y,grad_outputs=torch.ones_like(u_y),retain_graph=True,create_graph=True)[0]
        u = u.detach().cpu().numpy()
        u_x = u_x.detach().cpu().numpy()
        u_y = u_y.detach().cpu().numpy()
        u_xx = u_xx.detach().cpu().numpy()
        u_xy = u_xy.detach().cpu().numpy()
        u_yy = u_yy.detach().cpu().numpy()
        f_pde = f_pde.detach().cpu().numpy()
        return u, u_x, u_y, u_xx, u_xy, u_yy, f_pde, v

## 求单位法向,并删除边界最后一点
def find_normal(x):
    n_unit = np.array(x, copy=True)
    for i in range(x.shape[0]):
        a = x[i, :] - x[i - 1, :]
        b = x[(i + 1) * (i != x.shape[0] - 1), :] - x[i, :]
        a_n = np.array(a, copy=True)
        a_n[0], a_n[1] = a[1], -a[0]
        b_n = np.array(b, copy=True)
        b_n[0], b_n[1] = b[1], -b[0]
        a_abs = (a_n[0] ** 2 + a_n[1] ** 2) ** 0.5
        b_abs = (b_n[0] ** 2 + b_n[1] ** 2) ** 0.5
        n = b_n / b_abs + a_n / a_abs
        n_abs = (n[0] ** 2 + n[1] ** 2) ** 0.5
        n_unit[i, :] = n / n_abs
    return n_unit

# 将边界外的点删除,默认删除内部的点
def delete_point(x_area,bc,d_inside = True):
    polygon = shapely.geometry.Polygon(bc)
    points = shapely.geometry.MultiPoint(x_area)
    flag = np.ones(x_area.shape[0], dtype = np.bool)
    for i in range(x_area.shape[0]):
        flag[i] = polygon.covers(points[i])
    ff = np.where(flag == d_inside)
    x_area = np.delete(x_area, ff, axis=0)
    return x_area

## 主函数
dx = dy = 0.02
dxdy = dx*dy
layers = [2]+[20]*6+[1]
d = 2.0
ub = np.array([d/2,d/2])
lb = np.array([-d/2,-d/2])
x = np.arange(lb[0]+dx/2,ub[0],dx)
y = np.arange(lb[1]+dx/2,ub[1],dy)
X,Y = np.meshgrid(x,y)
x_area = np.hstack((X.reshape(-1,1),Y.reshape(-1,1)))
angle_circle = np.arange(0,2*math.pi,0.02)
x_bc = d/2*np.cos(angle_circle)
y_bc = d/2*np.sin(angle_circle)
bc = np.vstack((x_bc,y_bc)).T
x_area = delete_point(x_area,bc,False)
n = find_normal(bc)
# y_angle = 30
# num = 20
# l = 2.713/np.linspace(1, 20, endpoint=True, num=num)
# H_all = np.empty((0,num))
# for j in range(1):
#     H = np.array([])
#     for i in range(num):
#         cos_angle = np.cos(y_angle * math.pi / 180)
#         model = PhysicsInformedNN(x_area, bc, n, cos_angle, dxdy, layers, d, l[i], ub, lb)
#         model.train()
#         u_max = model.predict(bc)[0]
#         u_min = model.predict(np.array([[0.0,0.0]]))[0]
#         H = np.append(H, np.average(u_max)*(i+1))
#         print('j: %d, i: %d' % (j, i))
#     H_all = np.append(H_all, np.reshape(H,(1,-1)) , axis = 0)
# #
# writer = pd.ExcelWriter('Example2_1.xlsx')
# data_1 = pd.DataFrame(H_all.T)
# data_1.to_excel(writer, 'sheet_1', float_format='%.6f', header=False, index=False)
# writer.save()
# writer.close()
R = 7.5
Bo = R**2/2.713**2
cos_angle = np.cos(30 * math.pi / 180)
model = PhysicsInformedNN(x_area, bc, n, cos_angle, dxdy, layers,Bo,ub,lb)
model.train()
u_max = model.predict(bc)[0] * R
delta = np.average(u_max)
print('delta: %5f' % delta)
torch.save(model.dnn.state_dict(),'circle_Y_angle_30_15mm.pt')

# 画图
u_max = model.predict(bc)[0]
u_min = model.predict(np.array([[0.0, 0.0]]))[0]
print('u_max: %7f, u_min: %7f' % (np.average(u_max), u_min))
x_bc_err = torch.tensor(bc[:, 0:1], requires_grad=True).double().to(device)
y_bc_err = torch.tensor(bc[:, 1:2], requires_grad=True).double().to(device)
_, u_xbc, u_ybc, _, _, _, _, _ = model.predict(bc)
bc_err = np.arccos((model.net_bc_natural(x_bc_err, y_bc_err).detach().cpu().numpy() + cos_angle)) * 180 / math.pi

X_p = np.concatenate([x_area,bc], axis=0)
u, u_x, u_y, u_xx, u_xy, u_yy, f_pde, v = model.predict(X_p)
fig, ax = plt.subplots()
ax.set_aspect('equal')
tcf = ax.scatter(X_p[:, 0], X_p[:, 1], s=20, c=u_xx[:, 0], cmap='RdBu_r')
fig.colorbar(tcf)
ax.set_title('dudxx')

fig11, ax11 = plt.subplots()
ax11.set_aspect('equal')
tcf11 = ax11.scatter(X_p[:, 0], X_p[:, 1], s=20, c=u_xy[:, 0], cmap='RdBu_r')
fig11.colorbar(tcf11)
ax11.set_title('dudxy')

fig11, ax11 = plt.subplots()
ax11.set_aspect('equal')
tcf11 = ax11.scatter(X_p[:, 0], X_p[:, 1], s=20, c=u_yy[:, 0], cmap='RdBu_r')
fig11.colorbar(tcf11)
ax11.set_title('dudyy')

fig11, ax11 = plt.subplots()
ax11.set_aspect('equal')
tcf11 = ax11.scatter(X_p[:, 0], X_p[:, 1], s=20, c=u_x[:, 0], cmap='RdBu_r')
fig11.colorbar(tcf11)
ax11.set_title('dudx')

fig22, ax22 = plt.subplots()
ax22.set_aspect('equal')
tcf22 = ax22.scatter(X_p[:, 0], X_p[:, 1], s=20, c=u_y[:, 0], cmap='RdBu_r')
fig22.colorbar(tcf22)
ax22.set_title('dudy')

fig1, ax1 = plt.subplots()
ax1.set_aspect('equal')
tcf1 = ax1.scatter(X_p[:, 0], X_p[:, 1], s=20, c=f_pde[:, 0],
                   cmap='RdBu_r')
fig1.colorbar(tcf1)
ax1.set_title('area_err')

fig3, ax3 = plt.subplots()
ax3.set_aspect('equal')
tcf3 = ax3.scatter(X_p[:, 0], X_p[:, 1], s=20, c=u[:, 0], cmap='RdBu_r')
fig3.colorbar(tcf3)
ax3.set_title('Hight')

plt.show()
# loss曲线
plt.figure()
plt.yscale('log')
plt.plot(np.array(model.loss)[:,0],np.array (model.loss)[:,1],ls="-",lw=2,label="loss")
plt.plot(np.array(model.loss)[:,0],np.array(model.loss)[:,2],ls="-",lw=2,label="loss_bc")
plt.plot(np.array(model.loss)[:,0],np.array(model.loss)[:,3],ls="-",lw=2,label="loss_f")
plt.plot(np.array(model.loss)[:,0],np.array(model.loss)[:,4],ls="-",lw=2,label="loss_v")
plt.legend()
plt.grid(linestyle=":")
plt.axvline(x=2000,c="b",ls="--",lw=2)
plt.xlim(0,np.array(model.loss)[-1,0])
plt.show()
