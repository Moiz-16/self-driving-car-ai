# AI for Self Driving Car

# Importing the libraries

import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

# Creating the architecture of the Neural Network

class Network(nn.Module):
    #initialises our  object as soon as we create an abject from the network class
    def __init__(self, input_size, nb_action):
        super(Network, self).__init__()
        self.input_size = input_size
        self.nb__action = nb_action
        # the 1st hidden layer, how many neurons in and many out to the second layer
        self.fc1 = nn.Linear(input_size, 30)
        # the 2nd layer, how many neurons in and many out to the second next layer (correlates to no of actions)
        self.fc2 = nn.Linear(30, nb_action)
        
    #The function that activate our neurons, performing forward propogation
    def forward(self, state):
        # x = the hidden neurons, we use our first full connection to get our hidden neurons then apply the activation function to them
        x = F.relu(self.fc1(state))
        q_values = self.fc2(x)
        return q_values
    
# Implementing experience replay

class ReplayMemory(object):
    
    def __init__(self, capacity):
        # the maximum number of transistions we want to have in our memory
        self.capacity = capacity
        # memory contains the last 100 transistions
        self.memory = []
        
    # adds new transitions into the memory and makes sure that the memory always has a certain number transition soted
    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]
            
    def sample(self, batch_size):
        #takes random samples from the memory with a fixed size of batch size
        samples = zip(*random.sample(self.memory, batch_size))
        return map(lambda x: Variable(torch.cat(x, 0)), samples)
    
# Implementing deep Q learning

class Dqn():
    
    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma
        self.reward_window = []
        self.model = Network(input_size, nb_action)
        self.memory = ReplayMemory(100000)
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001)
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        self.last_action = 0
        self.last_reward = 0
        
    # selects the right action at each time
    
    def select_action(self, state):
        probs = F.softmax(self.model(Variable(state, volatile = True))*50) #Temperature parameter = 7, adjusts the probabilities giving greater certainty for which action to play, the higher it is the more the car will move like a car and not an insect
        action = probs.multinomial(num_samples=1)
        return action.data[0,0]
        
    # Training te deep learning network through back propogation and forward propogation     
    
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        #what the neural network predicts
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        target = self.gamma*next_outputs + batch_reward
        td_loss = F.smooth_l1_loss(outputs, target)
        self.optimizer.zero_grad()
        td_loss.backward(retain_graph = True)
        self.optimizer.step()
        
    # updates everything that needs to be updated as soon as the AI enters a new state, such as the action, state and reward
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
    
    #compputes the mean of ther rewards in the reward window 
    def score(self):
         return sum(self.reward_window)/(len(self.reward_window)+1)
    
    def save(self):
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer' : self.optimizer.state_dict(),
                   }, 'last_brain.pth')
    
    def load(self):
        if os.path.isfile('last_brain.pth'):
            print("=> loading checkpoint... ")
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("done !")
        else:
            print("no checkpoint found...")
    
          

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        