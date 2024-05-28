import torch

class Swish(torch.nn.Module):
    """ The class implements the Swish activation function from
    https://arxiv.org/pdf/2005.03191.pdf
    given input x. Swish(x) = x / (1 + exp(beta * x))
    Arguments
    ---------
    beta: float
        Beta value.
    Example
    -------
    >>> x = torch.randn((8, 40, 120))
    >>> act = Swish()
    >>> x = act(x)
    """

    def __init__(self, beta=1):
        super().__init__()
        self.beta = beta
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        """Returns the Swished input tensor.
        Arguments
        ---------
        x : torch.Tensor
            Input tensor.
        """
        return x * self.sigmoid(self.beta * x)