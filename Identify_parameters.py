from collections import OrderedDict
import torch
import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.tri as tri
import shapely.geometry
import pandas as pd
import time

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
    def __init__(self, X_u_data, x_area, x_bc, n_bc, dxdy, layers,ub,lb):
        self.x_data = torch.tensor(X_u_data[:, 0:1], requires_grad=True).double().to(device)
        self.y_data = torch.tensor(X_u_data[:, 1:2], requires_grad=True).double().to(device)
        self.u_data = torch.tensor(X_u_data[:, 2:3], requires_grad=True).double().to(device)
        self.x = torch.tensor(x_area[:, 0:1], requires_grad=True).double().to(device)
        self.y = torch.tensor(x_area[:, 1:2], requires_grad=True).double().to(device)
        self.x_bc = torch.tensor(x_bc[:, 0:1], requires_grad=True).double().to(device)
        self.y_bc = torch.tensor(x_bc[:, 1:2], requires_grad=True).double().to(device)
        self.n_bcx = torch.tensor(n_bc[:, 0:1]).double().to(device)
        self.n_bcy = torch.tensor(n_bc[:, 1:2]).double().to(device)
        self.ub = torch.tensor(ub).double().to(device)
        self.lb = torch.tensor(lb).double().to(device)
        self.lambda_lamb = torch.tensor([0.0], requires_grad=True).double().to(device)
        self.lambda_lamb = torch.nn.Parameter(self.lambda_lamb)
        # 设置求解参数
        self.lambda_cos_angle = torch.tensor([0.0], requires_grad=True).double().to(device)
        self.lambda_Bo = torch.tensor([1.0], requires_grad=True).double().to(device)
        self.lambda_cos_angle = torch.nn.Parameter(self.lambda_cos_angle)
        self.lambda_Bo = torch.nn.Parameter(self.lambda_Bo)
        self.layers = layers
        self.dxdy = dxdy
        # 定义一个深度网络
        self.dnn = DNN(layers).to(device)
        self.dnn.register_parameter('lambda_lamb', self.lambda_lamb)
        self.dnn.register_parameter('lambda_cos_angle', self.lambda_cos_angle)
        self.dnn.register_parameter('lambda_Bo', self.lambda_Bo)

        self.optimizer_LBFGS = torch.optim.LBFGS(
            self.dnn.parameters(),
            lr=1.0,
            max_iter=15000,
            max_eval=15000,
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
            +(- u * self.lambda_Bo + self.lambda_lamb)
        v = sum(sum(u)) * self.dxdy
        return f,v

    def net_bc_natural(self, x, y):
        u = self.net_u(x, y)
        u_x = torch.autograd.grad(u, x,grad_outputs=torch.ones_like(u),retain_graph=True,create_graph=True)[0]
        u_y = torch.autograd.grad(u, y,grad_outputs=torch.ones_like(u),retain_graph=True,create_graph=True)[0]
        bc = (u_x*self.n_bcx+u_y*self.n_bcy)/(1 + u_x ** 2 + u_y ** 2) ** .5 - self.lambda_cos_angle
        return bc

    def Calculate_loss(self):
        bc = self.net_bc_natural(self.x_bc, self.y_bc)
        f, v = self.net_f(self.x, self.y)
        f_bc = self.net_f(self.x_bc, self.y_bc)[0]
        f = torch.cat((f, f_bc), dim=0)
        loss_bc = torch.mean(bc ** 2)
        loss_f = torch.mean(f ** 2)
        loss_v = (self.lambda_Bo*v - 2 * math.pi*self.lambda_cos_angle) ** 2
        loss_data = torch.mean((self.net_u(self.x_data, self.y_data) - self.u_data) ** 2)
        a = 3000/torch.mean((self.u_data) ** 2)
        loss = 5000 * loss_bc + 1000 * loss_f + 100 * loss_v + a * loss_data

        self.iter += 1
        if self.iter % 100 == 0:
            print(
                'Iter %d, Loss: %.5e, Loss_bc: %.5e, Loss_f: %.5e, Loss_v: %.5e, Loss_data: %.5e' % (
                    self.iter, loss.item(), loss_bc.item(), loss_f.item(), loss_v.item(), loss_data.item())
            )
            print(
                'lambda_lamb: %.5e,lambda_Bo: %.5e,lambda_cos_angle: %.5e,' % (
                    self.lambda_lamb, self.lambda_Bo, self.lambda_cos_angle)
            )
        self.loss.append([self.iter, loss.item(), loss_bc.item(), loss_f.item(), loss_v.item(), loss_data.item()])
        return loss

    def loss_func(self):
        self.optimizer_LBFGS.zero_grad()
        loss = self.Calculate_loss()
        loss.backward()
        return loss

    def train(self):
        self.dnn.train()
        # 先使用Adam预训练
        for i in range(1,5000):
            self.optimizer_adam.zero_grad()
            loss = self.Calculate_loss()
            loss.backward()
            self.optimizer_adam.step()
        # end = time.perf_counter()
        # print("运行时间为", round(end - start), 'seconds')
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

def random_in_circle(R,N):
    #R radius of circle,N means the number of points
    N = round(N)
    t = np.random.random(N) #生成一组随机数
    t2 = np.random.random(N) #生成第二组随机数
    r = np.sqrt(t)*R        #密度与半径有平方反比关系
    theta = t2*2*np.pi       #角度是均匀分布
    x = r * np.cos(theta)     #换算成直角坐标系
    y = r * np.sin(theta)
    return np.c_[x,y]

## 主函数
dx = dy = 0.02
dxdy = dx*dy
layers = [2] + [10] * 6 + [1]
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
# 正解
layers1 = [2] + [20] * 6 + [1]
net = DNN(layers1)
lambda_lamb = torch.nn.Parameter(torch.tensor([0.0]))
Data_volume =np.arange(5, 100, 10)
start = time.perf_counter()
num = np.shape(Data_volume)[0]
for k in [10]:
    for noise in [0.0]:
        Bo_all = np.empty((0, num))
        cos_angle_all = np.empty((0, num))
        for j in range(1):
            Bo = np.array([])
            cos_angle = np.array([])
            for i in range(num):
                net.register_parameter('lambda_lamb', lambda_lamb)
                net.load_state_dict(torch.load(f'circle_Y_angle_30_{k}mm.pt'))
                x_data = random_in_circle(1,Data_volume[i])
                points = torch.from_numpy(x_data).double()
                u = net(points)
                u = u.detach().numpy()
                u = u + noise*np.std(u)*np.random.randn(u.shape[0], u.shape[1])
                xu_data = np.hstack((x_data,u))
                # 画数据图
                # plt.figure()
                # tcf1=plt.scatter(x_data[:, 0], x_data[:, 1],s = 20,c = u,cmap='RdBu_r')
                # plt.colorbar(tcf1)
                # plt.xlim(-1, 1)
                # plt.ylim(-1, 1)
                # plt.show()
                model = PhysicsInformedNN(xu_data, x_area, bc, n, dxdy, layers,ub,lb)
                # model.train()
                print('k: %d, noise: %f, j: %d, i: %d' % (k,noise , j, Data_volume[i]))
                Bo = np.append(Bo,model.dnn.lambda_Bo.detach().cpu().numpy())
                cos_angle = np.append(cos_angle, model.dnn.lambda_cos_angle.detach().cpu().numpy())
                end = time.perf_counter()
                print("运行时间为", round(end - start), 'seconds')
            Bo_all = np.append(Bo_all, np.reshape(Bo,(1,-1)) , axis=0)
            cos_angle_all = np.append(cos_angle_all, np.reshape(cos_angle,(1,-1)) , axis=0)
        # writer = pd.ExcelWriter(f'算例3管径{k}噪声{noise}.xlsx')
        # data_1 = pd.DataFrame(Bo_all.T)
        # data_2 = pd.DataFrame(cos_angle_all.T)
        # data_1.to_excel(writer, 'sheet_1', float_format='%.6f', header=False, index=False)
        # data_2.to_excel(writer, 'sheet_2', float_format='%.6f', header=False, index=False)
        # writer.save()
        # writer.close()
# # 画图
# X_p = np.concatenate([x_area,bc], axis=0)
# u = model.predict(X_p)[0][:,0]
# x = X_p[:,0]
# y = X_p[:,1]
# triang = tri.Triangulation(x, y)
# # 高度
# fig1, ax1 = plt.subplots()
# ax1.set_aspect('equal')
# tcf1 = ax1.tripcolor(triang, u, shading='gouraud',cmap='RdBu_r')
# fig1.colorbar(tcf1)
# ax1.set_title('Higth')
#
# # 误差云图
# hight_ture = net(torch.from_numpy(X_p).double())[:,0]
# hight_ture = hight_ture.detach().numpy()
#
# fig3, ax3 = plt.subplots()
# ax3.set_aspect('equal')
# tcf3 = ax3.tripcolor(triang,hight_ture, shading='gouraud',cmap='RdBu_r')
# fig3.colorbar(tcf3)
# ax3.set_title('Hight_ture')
#
# fig5, ax5 = plt.subplots()
# ax5.set_aspect('equal')
# tcf5 = ax5.tripcolor(triang,((u - hight_ture)**2/hight_ture**2)**0.5, shading='gouraud',cmap='RdBu_r')
# fig5.colorbar(tcf5)
# ax5.set_title('R2 error')
# # loss曲线
# plt.figure()
# plt.yscale('log')
# plt.plot(np.array(model.loss)[:,0],np.array(model.loss)[:,1],ls="-",lw=2,label="loss")
# plt.plot(np.array(model.loss)[:,0],np.array(model.loss)[:,2],ls="-",lw=2,label="loss_bc")
# plt.plot(np.array(model.loss)[:,0],np.array(model.loss)[:,3],ls="-",lw=2,label="loss_f")
# plt.plot(np.array(model.loss)[:,0],np.array(model.loss)[:,4],ls="-",lw=2,label="loss_data")
# plt.plot(np.array(model.loss)[:,0],np.array(model.loss)[:,4],ls="-",lw=2,label="loss_V")
# plt.legend()
# plt.grid(linestyle=":")
# plt.axvline(x=1000,c="b",ls="--",lw=2)
# plt.xlim(0,np.array(model.loss)[-1,0])
# plt.show()