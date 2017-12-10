from gym_torcs import TorcsEnv
import numpy as np
#import cv2

img_dim = [64,64,3]
action_dim = 1
steps = 1000
batch_size = 32
nb_epoch = 100

def get_teacher_action(ob):
    steer = ob.angle*10/np.pi
    steer -= ob.trackPos*0.10
    return np.array([steer])

def img_reshape(input_img):
    _img = np.transpose(input_img, (1, 2, 0))
    _img = np.flipud(_img)
    _img = np.reshape(_img, (1, img_dim[0], img_dim[1], img_dim[2]))
    return _img

images_all = np.zeros((0, img_dim[0], img_dim[1], img_dim[2]))
actions_all = np.zeros((0,action_dim))
rewards_all = np.zeros((0,))

img_list = []
action_list = []
reward_list = []

env = TorcsEnv(vision=True, throttle=False)
ob = env.reset(relaunch=True)

print('Collecting data...')
for i in range(500):
    if i == 0:
        act = np.array([0.0])
    else:
        act = get_teacher_action(ob)

    if i%100 == 0:
        print(i)
    ob, reward, done, _ = env.step(act)
    img_list.append(ob.img)
    action_list.append(act)
    reward_list.append(np.array([reward]))

env.end()

print('Packing data into arrays...')

for img, act, rew in zip(img_list, action_list, reward_list):
    images_all = np.concatenate([images_all, img_reshape(img)], axis=0)
    actions_all = np.concatenate([actions_all, np.reshape(act, [1,action_dim])], axis=0)
    rewards_all = np.concatenate([rewards_all, rew], axis=0)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Convolution2D, MaxPooling2D, Concatenate
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam

#model from https://github.com/fchollet/keras/blob/master/examples/cifar10_cnn.py
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=images_all.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(action_dim))
model.add(Activation('tanh'))

model.compile(loss='mean_squared_error',
              optimizer=Adam(lr=1e-4),
              metrics=['mean_squared_error'])

images_all = images_all.astype('float32')
images_all /= 255

model.fit(images_all, actions_all,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              shuffle=True)

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.autograd import Variable
# import itertools

# class Policy(nn.Module):
#     def __init__(self):
#         super(Policy, self).__init__()
#         self.layer1 = nn.Sequential(
#             nn.Conv2d(3, 16, kernel_size=5, padding=2),
#             nn.BatchNorm2d(16),
#             nn.ReLU(),
#             nn.MaxPool2d(2))
#         self.layer2 = nn.Sequential(
#             nn.Conv2d(16, 32, kernel_size=5, padding=2),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.MaxPool2d(2))

#         hidden_size = 64 # determine this ...  from below
#         self.aff_mu = nn.Linear(hidden_size, hidden_size)
#         self.aff_log_std = nn.Linear(hidden_size, hidden_size)

#         self.aff2_mu = nn.Linear(hidden_size, 1)
#         self.aff2_log_std = nn.Linear(hidden_size, 1)

#         self.act = nn.ReLU()
#         self.final = nn.Tanh()

#     def forward(self, x):
#         out = self.layer1(x)
#         out = self.layer2(out)
#         out = out.view(out.size(0), -1)
#         print(out.size()) # you can determine hidden_size from here
#         mu = self.final(self.aff2_mu(self.act(self.aff_mu(out))))
#         log_std = self.final(self.aff2_log_std(self.act(self.aff_log_std(out))))

#         std = torch.exp(log_std)

#         return mu, log_std, std

# def normal_log_density(x, mean, log_std, std):
#     var = std.pow(2)
#     log_density = -(x - mean).pow(2) / (
#         2 * var) - 0.5 * math.log(2 * math.pi) - log_std
#     return log_density.sum(1, keepdim=True)


# images_var = Variable(torch.from_numpy(images_all))
# actions_var = Variable(torch.from_numpy(actions_all))

# policy_net = Policy()
# params = itertools.ifilter(
#     lambda x: x.requires_grad, policy_net.parameters()
# )

# # define the optimizer to use; currently use Adam
# opt = optim.Adam(
#     params, lr=1e-4, weight_decay=1e-3
# )

# # train for 10 times
# for i in range(10):
#     opt.zero_grad()
#     mean, log_std, std = policy_net(images_var)
#     loss = normal_log_density(actions_var, mean, log_std, std)
#     opt.step()


output_file = open('results.txt', 'w')

def DAgger(lam, t, ob):
    p = lam ** t
    tau = np.random.rand()
    if tau < p:
        return np.reshape(get_teacher_action(ob), [1,action_dim])
    else:
        return model.predict(img_reshape(ob.img).astype('float32')/255)

def SafeDAgger(tau, ob):
    nov_act = model.predict(img_reshape(ob.img).astype('float32')/255)
    exp_act = np.reshape(get_teacher_action(ob), [1,action_dim])
    diff = np.sqrt(np.square(nov_act - exp_act))
    if diff < tau:
        return nov_act, 1
    else:
        return exp_act, 0

def LinearCombo(lam, t, ob, threshold):
    beta = 1 - 0.5 * lam ** t
    nov_act = model.predict(img_reshape(ob.img).astype('float32')/255)
    exp_act = np.reshape(get_teacher_action(ob), [1,action_dim])
    predicted_act = beta * nov_act + (1-beta) * exp_act
    diff = predicted_act - exp_act

    if diff ** 2 > threshold ** 2:
        return exp_act + threshold if diff > 0 else exp_act - threshold
    else:
        return predicted_act

def LookAhead(tau, ob, prev_act, alpha):
    nov_act = model.predict(img_reshape(ob.img).astype('float32')/255)
    exp_act = np.reshape(get_teacher_action(ob), [1,action_dim])
    cos = nov_act * exp_act / np.sqrt(nov_act **2 * exp_act **2)
    include_prev = False
    if cos < alpha:
        include_prev = True
    diff = np.sqrt(np.square(nov_act - exp_act))
    if diff < tau:
        return nov_act, 1, include_prev
    else:
        return exp_act, 0, include_prev

def NN(ob, c):
    inp = img_reshape(ob.img).astype('float32')/255
    mean, log_std, std = policy_net(Variable(torch.from_numpy(inp)))
    c1 = c+1
    while c1 > c:
        c1 = np.random.randn()

    nov_act = mean.data[0] + c1 * std.data[0]
    return nov_act

def test():
    env = TorcsEnv(vision=True, throttle=False)
    ob = env.reset(relaunch=True)
    reward_sum = 0.0
    done = False

    count = 0
    while not done:
        act = model.predict(img_reshape(ob.img).astype('float32')/255)
        #print(act)
        count += 1
        ob, reward, done, _ = env.step(act)
        reward_sum += reward    
    env.end()
    print("Steps before crash: ", count, reward_sum)
    return count, reward_sum

#aggregate and retrain
dagger_itr = 5
lam = 0.9
T = 100

for itr in range(dagger_itr):
    ob_list = []

    env = TorcsEnv(vision=True, throttle=False)
    ob = env.reset(relaunch=True)
    reward_sum = 0.0

    nov_count = 0
    prev_act = np.array([0])
    prev_included = False
    prev_ob = ob
    for i in range(T):
        #act = DAgger(lam, i, ob)
        #act, is_nov = SafeDAgger(0.05, ob)
        act = LinearCombo(0.9, i, ob, 0.05)
        #act, is_nov, include_prev = LookAhead(0.08, ob, prev_act, 0.1)
        #if include_prev and not prev_included:
        #    ob_list.append(prev_ob)
        #
        #act = NN(ob, 0.5)
        #is_nov = 0 # in case of NN, we always append
        prev_act = act
        is_nov = 0
        nov_count += is_nov
        print(act)
        ob, reward, done, _ = env.step(act)

        if done:
            break

        reward_sum += reward
        print("step{} R{} tot_R{} done{} act{} nov\%{}".format(i, reward, reward_sum, done, str(act[0]), nov_count/float(i+1)))

        if is_nov == 0:
            ob_list.append(ob)
            prev_included = True
        else:
            prev_ob = ob
            prev_included = False

    print('Episode done ', itr, i, reward_sum)
    
    env.end()

    count, reward_sum_t = test()
    output_file.write('Number of Steps: %02d\t Reward: %0.04f\n'%(count, reward_sum_t))

    for ob in ob_list:
        images_all = np.concatenate([images_all, img_reshape(ob.img).astype('float32')/255], axis=0)
        actions_all = np.concatenate([actions_all, np.reshape(get_teacher_action(ob), [1,action_dim])], axis=0)

    # in NN case
    '''
    images_var = Variable(torch.from_numpy(images_all))
    actions_var = Variable(torch.from_numpy(actions_all))
    for i in xrange(10):
        opt.zero_grad()
        mean, log_std, std = policy_net(images_var)
        loss = normal_log_density(actions_var, mean, log_std, std)
        opt.step()
    '''
    model.fit(images_all, actions_all,
                  batch_size=batch_size,
                  nb_epoch=nb_epoch,
                  shuffle=True)