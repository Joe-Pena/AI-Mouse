import numpy
import random
import os # Use to save or load brain 
import torch
import torch.nn as nn
import torch.nn.functional as nnF
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

class Network(nn.Module):
    
    def __init__(self, input_size, nb_action):
        super(Network, self).__init__()
        self.input_size = input_size #No of input nodes
        self.nb_action = nb_action #No of actions (Qs)
        self.full_connection1 = nn.Linear(input_size, 30)#from input to hidden
        self.full_connection2 = nn.Linear(30, nb_action)#from hidden to output(Qs)
        #TODO: hidden layer has 30 nodes, adjust later!
        
    def forward(self, state): #forward propagation
        x_layers = nnF.relu(self.full_connection1(state)) #hidden layer
        q_values = self.full_connection2(x_layers) #q_value layer
        return q_values #left, straight, or right!
    
class ReplayMemory(object): 
    
    def __init__(self, capacity):
        self.capacity = capacity #memory capacity of actions, ~100
        self.memory = [] 
        
    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0] #keep memory at 100 actions
            
    def sample(self, batch_size):
        samples = zip(*random.sample(self.memory, batch_size))
        return map(lambda smpl: Variable(torch.cat(smpl, 0)), samples)
    
class Dqn():
    
    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma
        self.reward_window = []
        self.model = Network(input_size, nb_action)
        self.memory = ReplayMemory(100000)
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.01)
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        self.last_action = 0
        self.last_reward = 0
        
    def select_action(self, state):
        probs = nnF.softmax(self.model(Variable(state, volatile = True))*150) #Temp = 100
        action = probs.multinomial()
        return action.data[0,0]
    
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        target = self.gamma * next_outputs + batch_reward
        td_loss = nnF.smooth_l1_loss(outputs, target) #error, propagate back based on number
        self.optimizer.zero_grad()
        td_loss.backward(retain_variables = True)
        self.optimizer.step()
        
    def update(self, reward, new_signal):
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))
        action = self.select_action(new_state)
        if len(self.memory.memory) > 100:
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
            self.last_action = action
            self.last_state = new_state
            self.last_reward = reward
            self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        return action
    
    def score(self):
        return sum(self.reward_window)/(len(self.reward_window) + 1.)
    
    def save(self): #saves self.model and self.optimizer
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    }, 'last_brain.pth') #saves nn to last_brain.pth
    
    def load(self): #load the last brain
        if os.path.isfile('last_brain.pth'):
            print('=> loading Neural Network')
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print('Finished loading')
        else:
            print('No previous Neural Network found')