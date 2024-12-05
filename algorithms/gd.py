# viet ham tim gia tri x de phuong trinh co cuc tri
# y = 2x^2 + x
import matplotlib.pyplot as plt

def derivative(xo):
   return 4*xo + 1

def timX(xo, learning_rate, iter):
   x_history = []
   x_new = xo
   for i in range(iter):
       x_new -= derivative(xo)*learning_rate
       x_history.append(x_new)
       # if daoHam(x_new) < 0.0001:
       #     break
   return x_new, x_history

x_new, x_history = timX(-3,0.001,500)

def hienThiJ(x_history):
   J = []
   for k in x_history:
       y = 2*k*k + k
       J.append(y)
   return J

J = hienThiJ(x_history)

plt.scatter(x_history, J)
plt.xlabel("gia tri cua X")
plt.ylabel("gia tri cua y")
plt.show()

