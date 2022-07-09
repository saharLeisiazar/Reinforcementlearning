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
            if v.tag == 'iteration':
                X.append(v.simple_value)
            elif v.tag == 'Eval_AverageReturn':
                Y.append(v.simple_value)
            elif v.tag == 'Eval_StdReturn':
                Z.append(v.simple_value)
    return X, Y, Z

if __name__ == '__main__':
    import glob

    num = 3
    file_list = ['hw4_q5_cheetah_cem_4_cheetah-cs285-v0_08-07-2022_16-03-27','hw4_q5_cheetah_cem_2_cheetah-cs285-v0_08-07-2022_15-55-42', 'hw4_q5_cheetah_random_cheetah-cs285-v0_08-07-2022_16-03-31']
    label_list = ['CEM:4', 'CEM:2', 'Random']  #'ntu:100 ,ngsptu:1', , 'ntu:10 ,ngsptu:10'
    c = ['blue', 'red', 'green']
    plt.figure()

    for i in range(num): 
               
        logdir = '/home/sahar/RL/homework_fall2021/hw4/data/' + file_list[i] + '/events*'
    
        eventfile = glob.glob(logdir)[0]

        X, Y, Z = get_section_results(eventfile)
#    for i, (x, y) in enumerate(zip(X, Y)):
#        print('Iteration {:d} | Train steps: {:d} | Return: {}'.format(i, int(x), y))

        plt.plot(X, Y, linestyle='-', label=label_list[i], c= c[i]) #, ecolor='lightgray'
        
    plt.xlabel("Number of itteration")
    plt.ylabel("Eval_AverageReturn")
    plt.legend()
    plt.savefig("/home/sahar/RL/homework_fall2021/hw4/results/Q4/CEM.png")  
    print('done')
    