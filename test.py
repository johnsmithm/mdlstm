import from mdlstm *
from random import randint
import numpy as np

def main():
    """
    count the ones from a matrix of zeros
    """
    graph = tf.Graph()
    with graph.as_default():
        #batch size, height, width, chanels
        input_data =  tf.placeholder(tf.float32, [2,4,6,1])
        nr =  tf.placeholder(tf.float32)
        
        sh = [2,2]
        out = tanAndSum(20,input_data,'l1',[2,2])
        
        outputs = tf.reshape(out, [-1, 20])
        weights = {
            'out': tf.Variable(tf.random_normal([20, 2]))
        }
        biases = {
            'out': tf.Variable(tf.random_normal([2]))
        }

        tt = tf.matmul(outputs, weights['out']) + biases['out']

        s = tf.reduce_mean(tt)
        cost = (s-nr)*(s-nr)
        
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

    
    with tf.Session(graph=graph) as session:
        tf.global_variables_initializer().run()
        for i in range(10000):
            #generate data for training
            dd = np.zeros([2,4,6,1]).astype('float')
            nn = randint(1,9)
            for k in range(2):
                for j in range(nn):
                    w = randint(0,5)
                    h = randint(0,3)
                    nr1 = 0
                    while dd[k,h,w,0] !=0.0 and nr1<10000:
                        nr1 = nr1 + 1
                        w = randint(0,5)
                        h = randint(0,3)
                    assert nr1 < 9098
                    dd[k,h,w,0] = 1.0
            nn = float(nn)     

            c,_ = session.run([cost,optimizer],{input_data:dd,nr:nn})
            if i%1000==0:
                print ("iteration: ",i,' cost:',c)
main()