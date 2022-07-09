import matplotlib.pyplot as plt 

plt.figure()
plt.plot(1, -36.89, marker='o', markersize=20, label='Eval_AverageReturn')
plt.plot(1, -167.098, marker='o', markersize=20, label='Train_AverageReturn')
plt.legend()

plt.savefig("/home/sahar/RL/homework_fall2021/hw4/results/Q2/plot.png")

print('done')
