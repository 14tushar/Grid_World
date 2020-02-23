import numpy as np
import gym
import grid_world
import tensorflow as tf

class DQN:
    def __init__(self, obs_size, n_actions, learning_rate=0.001, name='DQN'):
        with tf.variable_scope(name):
            #Variables_Required
            self.state = tf.placeholder(tf.float32, shape=(None,obs_size), name='state')
            self.action = tf.placeholder(tf.int32, shape=(None,), name='action')
            self.target_Q = tf.placeholder(tf.float32, shape=(None,), name='target')
            #Neural_Network
            self.layer1 = tf.layers.dense(self.state, 16, activation=tf.nn.relu, name='Layer_1')
            self.layer2 = tf.layers.dense(self.layer1, 16, activation=tf.nn.relu, name='Layer_2')
            self.output = tf.layers.dense(self.layer2, n_actions, activation=tf.nn.softmax, name='Layer_3')
            #Q-value Determination
            self.acted = tf.one_hot(self.action, n_actions, 1.0, 0.0, name='acted')
            self.Q = tf.reduce_sum(self.output*self.acted, name='Q')
            #Optimizing Loss
            self.loss = tf.reduce_mean(tf.square(self.target_Q - self.Q, name='loss_train'))
            self.opt = tf.train.AdamOptimizer(learning_rate).minimize(self.loss, name='adam_optim')

env = gym.make('PuddleWorld-v1')
obs_size = env.observation_space.shape[0]
n_actions = env.action_space.n
mainQN = DQN(obs_size, n_actions)

gamma=0.9
epsilon = 0.1

sess = tf.Session()
sess.run(tf.global_variables_initializer())

st = env.reset()
re=0
for _ in range(1000000):
    if np.random.random() < epsilon:
        a = env.action_space.sample()
    else:
        a = np.argmax(sess.run(mainQN.output,feed_dict={mainQN.state:np.array([st])}))
    st_next, r, done = env.step(a)
    re+=r
    if done:
        next_q_max = 0
    else:
        next_q_max = np.max(sess.run(mainQN.output,feed_dict={mainQN.state:np.array([st_next])}))
    target = r + gamma*next_q_max
    loss, _ = sess.run([mainQN.loss, mainQN.opt],feed_dict={mainQN.state:np.array([st]),mainQN.action:np.array([a]),mainQN.target_Q:np.array([target])})
    if done:
        st = env.reset()
        print('REWARD---------',re)
        print('LOSS',loss)
        re=0
    else:
        st = st_next

sess.close()
#kernel_initializer=tf.contrib.layers.xavier_initializer(),
