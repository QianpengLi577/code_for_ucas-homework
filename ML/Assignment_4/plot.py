import numpy as np
import matplotlib.pyplot as plt



R1 = np.loadtxt('R1.csv',dtype=np.float32, delimiter=',')
R2 = np.loadtxt('R2.csv',dtype=np.float32, delimiter=',')
R3 = np.loadtxt('R3.csv',dtype=np.float32, delimiter=',')
R4 = np.loadtxt('R4.csv',dtype=np.float32, delimiter=',')
R5 = np.loadtxt('R5.csv',dtype=np.float32, delimiter=',')
R6 = np.loadtxt('R6.csv',dtype=np.float32, delimiter=',')

x = np.linspace(0 , R1.shape[1], R1.shape[1])

plt.figure(figsize=(10,6))
plt.title('R1 result')
for i in range(R6.shape[0]):
    plt.plot(x,R6[i,:],label=str(i),ls=":")
# plt.plot(x,R2[4,:],label=str(0),ls=":")
plt.legend()
plt.xlabel('train step')
plt.ylabel('average reward')
plt.show()