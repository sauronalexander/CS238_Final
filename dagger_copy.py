from gym_torcs import TorcsEnv
import numpy as np
import cv2

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
for i in range(100):
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

# the neural network for discriminator
'''
cond = Input(batch_shape(batch_size, 1))
cnn = Sequential()
cnn.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=images_all.shape[1:]))
cnn.add(Activation('relu'))
cnn.add(Conv2D(32, (3, 3)))
cnn.add(Activation('relu'))
cnn.add(MaxPooling2D(pool_size=(2, 2)))
cnn.add(Dropout(0.5))

cnn.add(Conv2D(64, (3, 3), padding='same'))
cnn.add(Activation('relu'))
cnn.add(Conv2D(64, (3, 3)))
cnn.add(Activation('relu'))
cnn.add(MaxPooling2D(pool_size=(2, 2)))
cnn.add(Dropout(0.5))

cnn.add(Flatten())
cnn.add(Dense(10))

merged = Concatenate([cnn, cond])
result = Dense(1, activation='sigmoid')(merged)

result.compile(
    optimizer=Adam(lr=1e-4), 
    loss='mean_squared_error',
    metrics=['mean_squared_error'])

result.fit([images_all, actions_all], 
    batch_size=batch_size,
    nb_epoch=nb_epoch,
    shuffle=True,)

'''

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

def NN(ob):
    nov_act = model.predict(img_reshape(ob.img).astype('float32')/255)
    exp_act = np.reshape(get_teacher_action(ob), [1,action_dim])

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
    env.end()
    print("Steps before crash: ", count)

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
        #act = LinearCombo(0.9, i, ob, 0.05)
        #act, is_nov, include_prev = LookAhead(0.08, ob, prev_act, 0.1)
        #if include_prev and not prev_included:
        #    ob_list.append(prev_ob)
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
    output_file.write('Number of Steps: %02d\t Reward: %0.04f\n'%(i, reward_sum))
    env.end()

    test()

    for ob in ob_list:
        images_all = np.concatenate([images_all, img_reshape(ob.img).astype('float32')/255], axis=0)
        actions_all = np.concatenate([actions_all, np.reshape(get_teacher_action(ob), [1,action_dim])], axis=0)

    model.fit(images_all, actions_all,
                  batch_size=batch_size,
                  nb_epoch=nb_epoch,
                  shuffle=True)