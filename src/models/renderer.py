import torch
import torch.nn as nn

from models.positional_encoding import PositionalEncoding


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias.data)


class LinearRelu(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(LinearRelu, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(in_channels, out_channels, bias=bias),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layer(x)


class Renderer(nn.Module):

    def __init__(
            self,
            x_encoding_frequencies=10,
            volume_encoding_channels=8,
            image_colour_channels=3,
            view_count=4,
            hidden_layer_neurons=256,
            hidden_layer_count=6,
            skip_connection_index=4,
            direction_encoding_frequencies=9,
    ):
        """
        Construct a new renderer following the spec in MVSNeRF paper.

        :param x_encoding_frequencies: Number of frequencies in positional encoding of x
        :param volume_encoding_channels: Number of features in neural encoding volume
        :param image_colour_channels: Number of colour channels in input image
        :param view_count: Total number of views (sources + reference). Used for calculating pixel colour vector length
        :param hidden_layer_neurons: Number of neurons in hidden fully connected layers
        :param hidden_layer_count: Number of hidden layers in network
        :param skip_connection_index: Index of hidden layers to add skip connection
        :param direction_encoding_frequencies: Number of frequencies in positional encoding of direction vector
        """
        super(Renderer, self).__init__()

        # positional encoding for novel view camera position and direction
        self.x_positional_encoding = PositionalEncoding(x_encoding_frequencies, concat_inputs=True)
        self.d_positional_encoding = PositionalEncoding(direction_encoding_frequencies, concat_inputs=False)

        # weight layer for cost volume and colour data
        self.lr_zero = LinearRelu(volume_encoding_channels + view_count * image_colour_channels, hidden_layer_neurons)

        # weight layer for viewing direction
        x_encoded_dimension = (x_encoding_frequencies + 1) * 3
        self.lr_one = LinearRelu(x_encoded_dimension, hidden_layer_neurons)

        # weight layers 2, ..., hidden_layer_count
        lr_i = []
        for layer_index in range(hidden_layer_count - 1):
            if (layer_index + 2) == skip_connection_index:
                self.skip_connection_index = layer_index
                lr_i.append(LinearRelu(x_encoded_dimension + hidden_layer_neurons, hidden_layer_neurons))
            else:
                lr_i.append(LinearRelu(hidden_layer_neurons, hidden_layer_neurons))
        self.lr_i = nn.ModuleList(lr_i)

        # volume density
        self.density_out = LinearRelu(hidden_layer_neurons, 1)

        # final two layers combine direction and hidden layer outputs to predict colour
        self.lr_direction = LinearRelu(3 * x_encoding_frequencies + hidden_layer_neurons, hidden_layer_neurons)
        self.colour_out = LinearRelu(hidden_layer_neurons, image_colour_channels)

    def init_weights(self):
        self.lr_zero.apply(weights_init)
        self.lr_one.apply(weights_init)
        self.lr_i.apply(weights_init)
        self.density_out.apply(weights_init)
        self.colour_out.apply(weights_init)

    def forward(self, x, d, f, c):
        """
        Forward pass through renderer network.

        :param x: Novel view position
        :param d: Novel view direction
        :param f: Volume encoding
        :param c: Colour data of images at point
        :return: Predicted colour and density
        """
        # PE_0
        x = self.x_positional_encoding(x)

        # LR_0
        lr_0 = self.lr_zero(torch.cat([f, c], dim=1))

        # LR_1
        values = self.lr_one(x)

        # LR_2,...,hidden_layer_count
        for (layer_index, lr_i) in enumerate(self.lr_i):
            if layer_index == self.skip_connection_index:
                # add inputs for skip connection
                values = torch.cat([x, values], -1)

            values = torch.einsum('ik,jk->i', values, lr_0)
            values = lr_i(values)

        # density value
        sigma = self.density_out(values)

        # colour value
        d_positional_encoding = self.d_positional_encoding(d)
        values = self.lr_direction(torch.cat([values, d_positional_encoding], dim=1))
        values = self.colour_out(values)

        return values, sigma
