import glob
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy

def get_section_results(file):
    """
        requires tensorflow==1.12.0
    """
    X = []
    Y = []
    Z = []
    for e in tf.train.summary_iterator(file):
        for v in e.summary.value:
            if v.tag == 'Iteration':
                X.append(v.simple_value)
            elif v.tag == 'Eval_AverageReturn':
                Y.append(v.simple_value)
            elif v.tag == 'Eval_StdReturn':
                Z.append(v.simple_value)
    return X, Y, Z

if __name__ == '__main__':
    import glob

    num = 1
    file_list = ['q5_1_100_HalfCheetah-v2_28-06-2022_14-56-16']#, 'q4_1_100_CartPole-v0_28-06-2022_14-43-59', 'q4_10_10_CartPole-v0_28-06-2022_14-53-58']
    label_list = [ 'ntu:1 ,ngsptu:100']  #'ntu:100 ,ngsptu:1', , 'ntu:10 ,ngsptu:10'

    plt.figure()

    for i in range(num): 
               
        logdir = '/home/sahar/RL/homework_fall2021/hw3/data/' + file_list[i] + '/events*'
    
        eventfile = glob.glob(logdir)[0]

        X, Y, Z = get_section_results(eventfile)
#    for i, (x, y) in enumerate(zip(X, Y)):
#        print('Iteration {:d} | Train steps: {:d} | Return: {}'.format(i, int(x), y))

        plt.errorbar(X, Y, Z, linestyle='-', label=label_list[i], c=numpy.random.rand(3,), ecolor='lightgray')
        
    plt.xlabel("Number of itteration")
    plt.ylabel("Average Reward")
    plt.legend()
    plt.savefig("/home/sahar/RL/homework_fall2021/hw3/results/q5_halfchita.png")  
    print('done')