import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

def init_params(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)

class AtariACModel(nn.Module):
    def __init__(self, action_space, use_memory=False):
        super().__init__()
        self.use_memory = use_memory
        self.recurrent = use_memory
        
        # Atari CNN (similar to DQN paper)
        self.image_conv = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU()
        )
        
        # Output: 7×7×64 = 3136
        self.image_embedding_size = 3136
        
        # Memory
        if use_memory:
            self.memory_rnn = nn.LSTMCell(self.image_embedding_size, 512)
            embedding_size = 512
        else:
            embedding_size = self.image_embedding_size
            
        # Actor-critic heads
        self.actor = nn.Sequential(
            nn.Linear(embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, action_space.n)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        # Apply weight initialization
        self.apply(init_params)
    
    @property
    def memory_size(self):
        return 1024 if self.use_memory else 0
    
    
    def forward(self, obs, memory=None):
        # obs shape: (batch, 4, 84, 84)
        x = self.image_conv(obs)
        x = x.reshape(x.shape[0], -1)
        
        if self.use_memory:
            hidden = (memory[:, :512], memory[:, 512:])
            hidden = self.memory_rnn(x, hidden)
            embedding = hidden[0]
            memory = torch.cat(hidden, dim=1)
        else:
            embedding = x
        
        # Store activations for CB
        self.last_activations = [
            self.actor[1](self.actor[0](embedding)),
            self.critic[1](self.critic[0](embedding))
        ]
        
        dist = Categorical(logits=F.log_softmax(self.actor(embedding), dim=1))
        value = self.critic(embedding).squeeze(1)
        
        # Return 2 or 3 values depending on memory
        if self.use_memory:
            return dist, value, memory
        else:
            return dist, value