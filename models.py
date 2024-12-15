from torch import no_grad, stack
from torch.utils.data import DataLoader
from torch.nn import Module
import torch


"""
Functions you should use.
Please avoid importing any other torch functions or modules.
Your code will not pass if the gradescope autograder detects any changed imports
"""
from torch.nn import Parameter, Linear
from torch import optim, tensor, tensordot, empty, ones
from torch.nn.functional import cross_entropy, relu, mse_loss
from torch import movedim


class PerceptronModel(Module):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.

        In order for our autograder to detect your weight, initialize it as a 
        pytorch Parameter object as follows:

        Parameter(weight_vector)

        where weight_vector is a pytorch Tensor of dimension 'dimensions'

        
        Hint: You can use ones(dim) to create a tensor of dimension dim.
        """
        super(PerceptronModel, self).__init__()
        
        w_init = torch.ones(dimensions, dtype=torch.float32)
        self.w = Parameter(w_init.unsqueeze(0))


    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)

        The pytorch function `tensordot` may be helpful here.
        """
        return torch.tensordot(self.w, x, dims=([1],[1]))


    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        score = self.run(x)
        return 1 if score.item() >= 0 else -1



    def train(self, dataset):
        """
        Train the perceptron until convergence.
        You can iterate through DataLoader in order to 
        retrieve all the batches you need to train on.

        Each sample in the dataloader is in the form {'x': features, 'label': label} where label
        is the item we need to predict based off of its features.
        """        
        with no_grad():
            dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
            while True:
                error_count = 0
                for batch in dataloader:
                    x = batch['x']
                    y = batch['label']
                    prediction = self.get_prediction(x)
                    if prediction != y.item():
                        error_count += 1
                        self.w.data += (y.item() * x)
                if error_count == 0:
                    break

#===================================================

class RegressionModel(Module):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        super().__init__()
        self.layer1 = Linear(1, 200)
        self.layer2 = Linear(200, 200)
        self.layer3 = Linear(200, 1)




    def forward(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        x = relu(self.layer1(x))
        x = relu(self.layer2(x))
        x = self.layer3(x)
        return x  

    
    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a tensor of size 1 containing the loss
        """
        pred = self.forward(x)
        return mse_loss(pred, y)
    
  

    def train(self, dataset):
        """
        Trains the model.

        In order to create batches, create a DataLoader object and pass in `dataset` as well as your required 
        batch size. You can look at PerceptronModel as a guideline for how you should implement the DataLoader

        Each sample in the dataloader object will be in the form {'x': features, 'label': label} where label
        is the item we need to predict based off of its features.

        Inputs:
            dataset: a PyTorch dataset object containing data to be trained on
            
        """
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        optimizer = optim.Adam(self.parameters(), lr=0.002)

        for epoch in range(5000):
            total_loss = 0.0
            for batch in dataloader:
                bx, by = batch['x'], batch['label']
                optimizer.zero_grad()
                loss = self.get_loss(bx, by)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            if avg_loss < 0.015:
                break

#===================================================

class DigitClassificationModel(Module):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        super().__init__()
        self.layer1 = Linear(784, 200)
        self.layer2 = Linear(200, 100)
        self.layer3 = Linear(100, 10)

    def forward(self, x):
        # Standard PyTorch forward method
        return self.run(x)
    
    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a tensor with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        x = relu(self.layer1(x))
        x = relu(self.layer2(x))
        x = self.layer3(x)
        return x


    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a tensor with shape (batch_size x 784)
            y: a tensor with shape (batch_size x 10)
        Returns: a loss tensor that depends on both x and y
        """
        logits = self.forward(x)
        return cross_entropy(logits, y)

        

    def train(self, dataset):
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
        optimizer = optim.Adam(self.parameters(), lr=0.001)

        for _ in range(50):
            for batch in dataloader:
                bx, by = batch['x'], batch['label']
                optimizer.zero_grad()
                loss = self.get_loss(bx, by)
                loss.backward()
                optimizer.step()

            # Check validation accuracy
            val_acc = dataset.get_validation_accuracy()
            if val_acc > 0.98:
                break
