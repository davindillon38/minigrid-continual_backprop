import torch
from math import sqrt
class ContinualBackprop:
    """
    Continual Backpropagation for Actor-Critic models
    Adapted from https://github.com/shibhansh/loss-of-plasticity
    """

    def __init__(self, model, optimizer, 
                 replacement_rate=1e-5, decay_rate=0.99,
                 maturity_threshold=100, device='cpu'):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.total_replaced = 0

        # Hyperparameters
        self.replacement_rate = replacement_rate
        self.decay_rate = decay_rate
        self.maturity_threshold = maturity_threshold

        # Track utility and age for actor and critic hidden layers
        self.layers = [model.actor[0], model.critic[0]] # The Linear(64) layers
        self.layer_names = ['actor', 'critic']
        
        # Initialize tracking
        self.util = []
        self.ages = []
        self.mean_activations = []

        for layer in self.layers:
            num_neurons = layer.out_features
            self.util.append(torch.zeros(num_neurons).to(device))
            self.ages.append(torch.zeros(num_neurons).to(device))
            self.mean_activations.append(torch.zeros(num_neurons).to(device))

        # Compute initialization bounds for each layer
        self.bounds = []
        for layer in self.layers:
            # Kaiming initialization bound for ReLU/Tanh
            bound = torch.nn.init.calculate_gain('tanh') * sqrt(3 / layer.in_features)
            self.bounds.append(bound)

    def update_utility(self, layer_idx, activations):
        """
        Update utility scores for neurons in a layer 
        activations: (batch_size, num_neurons) tensor 
        of post-activation values
        """
        with torch.no_grad():
            # Decay existing utility
            self.util[layer_idx] *= self.decay_rate
            
            # Update mean activations (for bias correction)
            self.mean_activations[layer_idx] *= self.decay_rate
            self.mean_activations[layer_idx] += (1 - self.decay_rate) * activations.mean(dim=0)

            # Calculate contribution to utility
            current_layer = self.layers[layer_idx]

            # Get next layer (actor[2] or critic[2] - the output layers)
            if layer_idx == 0: # actor
                next_layer = self.model.actor[2]
            else: # critic
                next_layer = self.model.critic[2]

            # Contribution = /activation/ * /outgoing_weight_magnitude/
            output_weight_mag = next_layer.weight.data.abs().mean(dim=0)
            new_util = output_weight_mag * activations.abs().mean(dim=0)

            self.util[layer_idx] += (1 - self.decay_rate) * new_util
            
            # Bias correction (like Adam)
            bias_correction = 1 - self.decay_rate ** self.ages[layer_idx]
            bias_correction = torch.clamp(bias_correction, min=1e-8) # Avoid division by zero
            self.bias_corrected_util = self.util[layer_idx] / bias_correction
        
    def find_neurons_to_replace(self, layer_idx):
        """Find low-utility mature neurons to replace"""
        # Increment age
        self.ages[layer_idx] += 1
        
        # Find eligible (mature) neurons
        eligible = torch.where(self.ages[layer_idx] > self.maturity_threshold)[0]

        if len(eligible) == 0:
            return torch.empty(0, dtype=torch.long).to(self.device)
        
        # Calculate number to replace
        num_to_replace = self.replacement_rate * len(eligible)

        # Handle fractional replacements probabilistically
        if num_to_replace < 1:
            if torch.rand(1).item() <= num_to_replace:
                num_to_replace = 1
            else:
                return torch.empty(0, dtype=torch.long).to(self.device)
            
        num_to_replace = int(num_to_replace)

        # Find lowest utility neurons among eligible
        utilities = self.bias_corrected_util[eligible]
        lowest_util_indices = torch.topk(-utilities, min(num_to_replace, len(utilities)))[1]

        return eligible[lowest_util_indices]
    

    def reinitialize_neurons(self, layer_idx, neuron_indices):
        if len(neuron_indices) == 0:
            return
        
        self.total_replaced += len(neuron_indices)
        # LOG UTILITIES BEFORE REPLACEMENT
        print(f"\nReplacing {len(neuron_indices)} neurons in {self.layer_names[layer_idx]}")
        print(f"  Total replaced so far: {self.total_replaced}")
        print(f"  Utility range: [{self.util[layer_idx].min():.6f}, {self.util[layer_idx].max():.6f}]")
        print(f"  Replacing utilities: {self.util[layer_idx][neuron_indices].cpu().numpy()}")
        print(f"  Ages: {self.ages[layer_idx][neuron_indices].cpu().numpy()}")
        


        current_layer = self.layers[layer_idx]
        # Fixed: access through self.model
        next_layer = self.model.actor[2] if layer_idx == 0 else self.model.critic[2]
        
        bound = self.bounds[layer_idx]
        
        current_layer.weight.data[neuron_indices, :] = \
            torch.empty(len(neuron_indices), current_layer.in_features).uniform_(-bound, bound).to(self.device)
        current_layer.bias.data[neuron_indices] = 0
        
        next_layer.weight.data[:, neuron_indices] = 0
        
        self.util[layer_idx][neuron_indices] = 0
        self.ages[layer_idx][neuron_indices] = 0
        self.mean_activations[layer_idx][neuron_indices] = 0

    def step(self):
        """
        Perform one continual backprop step
        Call this after optimizer.step() in training loop
        """
        if not hasattr(self.model, 'last_activations'):
            return # No activations captured yet
        
        activations = self.model.last_activations

        for layer_idx in range(len(self.layers)):
            # Update utility based on recent activations
            self.update_utility(layer_idx, activations[layer_idx])

            # Find neurons to replace
            neurons_to_replace = self.find_neurons_to_replace(layer_idx)

            # Reinitialize those neurons
            self.reinitialize_neurons(layer_idx, neurons_to_replace)

            if len(neurons_to_replace) > 0:
                print(f'Replaced {len(neurons_to_replace)} neurons in {self.layer_names[layer_idx]} layer')

