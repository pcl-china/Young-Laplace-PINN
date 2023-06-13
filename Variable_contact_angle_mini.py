from collections import OrderedDict
import torch
import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.tri as tri
import shapely.geometry
import pandas as pd
import time

# 定义子网络列表
def NN_list(layers):
    depth = len(layers) - 1
    activation = torch.nn.Tanh
    layer_list = list()
    for i in range(depth - 1):
        layer_list.append(('layer_%d' % i, torch.nn.Linear(layers[i], layers[i + 1])))
        layer_list.append(('activation_%d' % i, activation()))
    layer_list.append(
        ('layer_%d' % (depth - 1), torch.nn.Linear(layers[-2], layers[-1])))
    layerDict = OrderedDict(layer_list)
    return layerDict

# 定义网络
class DNN(torch.nn.Module):
    def __init__(self, layers):
        super(DNN, self).__init__()
        self.layers = torch.nn.Sequential(NN_list(layers)).double()

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
    def __init__(self, x_area, x_bc, n_bc, cos_angle, dxdy, v0, layers,Bo,ub,lb):
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
        self.dxdy = dxdy
        self.v0 = v0
        # 定义一个深度网络
        self.dnn = DNN(layers).to(device)
        self.dnn.register_parameter('lambda_lambda_lamb', self.lambda_lamb)


        self.optimizer_LBFGS = torch.optim.LBFGS(
            self.dnn.parameters(),
            lr=1.0,
            max_iter=150000,
            max_eval=150000,
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
        u = self.dnn(X)[:,0:1]
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
        bc = (u_x*self.n_bcx+u_y*self.n_bcy)/(1 + u_x ** 2 + u_y ** 2) ** .5 - torch.unsqueeze(self.cos_angle, 1)
        return bc

    def Calculate_loss(self):
        bc = self.net_bc_natural(self.x_bc, self.y_bc)
        f_pde, v = self.net_f(self.x, self.y)
        f_bc = self.net_f(self.x_bc, self.y_bc)[0]
        f = torch.cat((f_pde, f_bc), dim=0)
        loss_bc = torch.mean(bc ** 2)
        loss_f = torch.mean(f ** 2)
        loss_v = (v - self.v0) ** 2
        loss = 10000 * loss_bc + 1000 * loss_f + 100 * loss_v
        self.iter += 1
        if self.iter % 100 == 0:
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
        for i in range(1,10000):
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

start = time.perf_counter()
## 主函数
dx = dy = 0.02
dxdy = dx*dy
d = 2.0
Bo = 1.0
v0 = 0.0
layers = [2]+[20]*8+[1]
ub = np.array([d / 2, d / 2])
lb = np.array([-d / 2, -d / 2])
x = np.arange(lb[0] + dx / 2, ub[0], dx)
y = np.arange(lb[1] + dx / 2, ub[1], dy)
X, Y = np.meshgrid(x, y)
x_area = np.hstack((X.reshape(-1, 1), Y.reshape(-1, 1)))
angle_circle = np.linspace(0, 2.0 * np.pi, endpoint=False, num=628)
x_bc = d / 2 * np.cos(angle_circle)
y_bc = d / 2 * np.sin(angle_circle)
bc = np.vstack((x_bc, y_bc)).T
x_area = delete_point(x_area, bc, False)
n = find_normal(bc)

# jiaodu = angle_circle
# cos_angle = np.ones(jiaodu.shape)*np.cos(60*math.pi/180)
# cos_angle[int(jiaodu.shape[0]/8):int(2*jiaodu.shape[0]/8)] = np.cos(60*math.pi/180)
# cos_angle[2*int(jiaodu.shape[0]/8):int(3*jiaodu.shape[0]/8)] = np.cos(120*math.pi/180)
# cos_angle[3*int(jiaodu.shape[0]/8):int(4*jiaodu.shape[0]/8)] = np.cos(120*math.pi/180)
# cos_angle[4*int(jiaodu.shape[0]/8):int(5*jiaodu.shape[0]/8)] = np.cos(60*math.pi/180)
# cos_angle[5*int(jiaodu.shape[0]/8):int(6*jiaodu.shape[0]/8)] = np.cos(60*math.pi/180)
# cos_angle[6*int(jiaodu.shape[0]/8):int(7*jiaodu.shape[0]/8)] = np.cos(120*math.pi/180)
# cos_angle[7*int(jiaodu.shape[0]/8):int(8*jiaodu.shape[0]/8)] = np.cos(120*math.pi/180)

# cos_angle = np.cos((np.sin(4 * angle_circle) * 30 + 90) * math.pi / 180)
# cos_angle = (np.cos(angle_circle)/2+np.double(angle_circle < math.pi)-0.5)

cos_angle = np.cos(2*angle_circle)/2
cos_angle[0:int(1*angle_circle.shape[0]/4)] =cos_angle[0:int(1*angle_circle.shape[0]/4)] + 0.5
cos_angle[int(1*angle_circle.shape[0]/4):int(2*angle_circle.shape[0]/4)] =cos_angle[int(1*angle_circle.shape[0]/4):int(2*angle_circle.shape[0]/4)] - 0.5
cos_angle[int(2*angle_circle.shape[0]/4):int(3*angle_circle.shape[0]/4)] =cos_angle[int(2*angle_circle.shape[0]/4):int(3*angle_circle.shape[0]/4)] + 0.5
cos_angle[int(3*angle_circle.shape[0]/4):int(4*angle_circle.shape[0]/4)] =cos_angle[int(3*angle_circle.shape[0]/4):int(4*angle_circle.shape[0]/4)] - 0.5
cos_angle = cos_angle / 2

# cos_angle = np.ones(angle_circle.shape)
# cos_angle[0:int(angle_circle.shape[0]/2)] = np.linspace(np.cos(135*math.pi/180), np.cos(120*math.pi/180),
#                                                                          endpoint=True, num=int(angle_circle.shape[0]/2))
# cos_angle[int(angle_circle.shape[0]/2):int(2*angle_circle.shape[0]/2)] = np.linspace(np.cos(45*math.pi/180), np.cos(60*math.pi/180),
#                                                                          endpoint=True, num=int(2*angle_circle.shape[0]/2)-int(angle_circle.shape[0]/2))


model = PhysicsInformedNN(x_area, bc, n, cos_angle, dxdy, v0, layers,Bo,ub,lb)
# model.train()
x_bc_err = torch.tensor(bc[:, 0:1], requires_grad=True).double().to(device)
y_bc_err = torch.tensor(bc[:, 1:2], requires_grad=True).double().to(device)
u_x = model.predict(bc)[1]
u_y = model.predict(bc)[2]
bc_err = model.net_bc_natural(x_bc_err, y_bc_err).detach().cpu().numpy()
         # / (1 + u_x ** 2 + u_y ** 2) ** 0.5

# writer = pd.ExcelWriter(f'结果.xlsx')
# data_1 = pd.DataFrame(cos_angle.T)
# data_2 = pd.DataFrame(bc_err)
# data_3 = pd.DataFrame(angle_circle)
# data_1.to_excel(writer, 'sheet_1', float_format='%.6f', header=False, index=False)
# data_2.to_excel(writer, 'sheet_2', float_format='%.6f', header=False, index=False)
# data_3.to_excel(writer, 'sheet_3', float_format='%.6f', header=False, index=False)
# writer.save()
# writer.close()

X_p = np.concatenate([x_area,bc], axis=0)
x = X_p[:,0]
y = X_p[:,1]
x_area_err = torch.tensor(X_p[:, 0:1], requires_grad=True).double().to(device)
y_area_err = torch.tensor(X_p[:, 1:2], requires_grad=True).double().to(device)
area_err = model.net_f(x_area_err, y_area_err)[0].detach().cpu().numpy()
du_dx = model.predict(X_p)[1]
du_dy = model.predict(X_p)[2]

triang = tri.Triangulation(x, y)
fig1, ax1 = plt.subplots()
ax1.set_aspect('equal')
tcf1 = ax1.tripcolor(triang,area_err[:, 0], shading='gouraud',cmap='RdBu_r')
                      # ,vmin=-0.03, vmax=0.03)
# tcf1 = ax1.scatter(X_p[:, 0], X_p[:, 1], s=20, c=area_err[:, 0]/(1 + du_dx[:, 0] ** 2 + du_dy[:, 0] ** 2) ** 1.0,cmap='RdBu_r'
#                    ,vmin=-0.08, vmax=0.08)
plt.axis('off')
fig1.colorbar(tcf1)
ax1.set_title('area_err')

fig2, ax2 = plt.subplots()
ax2.set_aspect('equal')
tcf2 = ax2.tripcolor(triang,du_dx[:, 0], shading='gouraud',cmap='RdBu_r')
                     # ,vmin=-0.02, vmax=0.02)
plt.axis('off')
fig2.colorbar(tcf2)
ax2.set_title('du_dx')

fig3, ax3 = plt.subplots()
ax3.set_aspect('equal')
tcf3 = ax3.tripcolor(triang,du_dy[:, 0], shading='gouraud',cmap='RdBu_r')
                     # ,vmin=-0.02, vmax=0.02)
plt.axis('off')
fig3.colorbar(tcf3)
ax3.set_title('du_dx')

u = model.predict(X_p)[0]
fig1, ax1 = plt.subplots()
ax1.set_aspect('equal')
tcf1 = ax1.tripcolor(triang, u[:, 0], shading='gouraud',cmap='RdBu_r')
# ax1.tricontour(triang, u, colors='k')
levels = np.linspace(u.min(), u.max(), 10)
tcf2 = ax1.tricontour(triang, u[:, 0],levels,
                      colors=['0.0', '0.0'],
                      linewidths=1.0)
ax1.clabel(tcf2, fmt='%.4f', colors='k', fontsize=14)
cbar = fig1.colorbar(tcf1)
cbar.add_lines(tcf2)
plt.axis('off')
# ax1.set_title('Higth')
# 画图
end = time.perf_counter()
print("运行时间为", round(end - start), 'seconds')