import torch
import torch.nn as nn
from kornia.nerf.positional_encoder import PositionalEncoder


class PositionalEncoding(PositionalEncoder):
    def __init__(self, num_dims, num_freqs, log_space=False):
        super(PositionalEncoding, self).__init__(num_dims, num_freqs, log_space)

    @property
    def num_encoded_dims(self):
        """Number of encoded dimensions."""
        return self._num_encoded_dims + self._num_dims

    def forward(self, x):
        return torch.cat(
            (x, super().forward(x)), dim=-1
        )


def weights_init(module):
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight)

        if module.bias is not None:
            nn.init.zeros_(module.bias)


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
            view_count=3,
            hidden_layer_neurons=256,
            hidden_layer_count=6,
            skip_connection_index=4,
            direction_encoding_frequencies=4,
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
        # TODO : 3 / 63
        self.x_positional_encoding = PositionalEncoder(3, x_encoding_frequencies)
        # TODO: 3 / 27
        self.d_positional_encoding = PositionalEncoder(3, direction_encoding_frequencies)

        # weight layer for cost volume and colour data
        self.lr_zero = LinearRelu(volume_encoding_channels + view_count * image_colour_channels, hidden_layer_neurons)

        # weight layer for viewing direction
        # TODO: 63
        #x_encoded_dimension = (2 * x_encoding_frequencies + 1) * 3
        self.lr_one = LinearRelu(self.x_positional_encoding.num_encoded_dims, hidden_layer_neurons)

        self.lr_two_four = nn.ModuleList([
            LinearRelu(hidden_layer_neurons, hidden_layer_neurons),
            LinearRelu(hidden_layer_neurons, hidden_layer_neurons),
            LinearRelu(hidden_layer_neurons, hidden_layer_neurons),
        ])

        self.lr_five = LinearRelu(self.x_positional_encoding.num_encoded_dims + hidden_layer_neurons, hidden_layer_neurons)
        self.lr_six = LinearRelu(hidden_layer_neurons, hidden_layer_neurons)
        self.lr_seven = LinearRelu(hidden_layer_neurons, hidden_layer_neurons)

        # volume density
        self.density_out = LinearRelu(hidden_layer_neurons, 1)

        # final two layers combine direction and hidden layer outputs to predict colour
        # TODO: 27
        d_encoded_dimension = (2 * direction_encoding_frequencies + 1) * 3
        self.lr_eight = LinearRelu(hidden_layer_neurons, hidden_layer_neurons)
        self.lr_direction = LinearRelu(self.d_positional_encoding.num_encoded_dims + hidden_layer_neurons, hidden_layer_neurons)
        self.colour_out = LinearRelu(hidden_layer_neurons, image_colour_channels)

    def init_weights(self):
        self.lr_zero.apply(weights_init)
        self.lr_one.apply(weights_init)
        self.lr_two_four.apply(weights_init)
        self.lr_five.apply(weights_init)
        self.lr_six.apply(weights_init)
        self.lr_seven.apply(weights_init)
        self.lr_eight.apply(weights_init)
        self.lr_direction.apply(weights_init)
        self.density_out.apply(weights_init)
        self.colour_out.apply(weights_init)

    def forward(self, view_positions, view_directions, volume_features, colours):
        # PE_0 : (batch_size_global, 63)
        view_positions = self.x_positional_encoding(view_positions)
        view_position_skip = view_positions

        # PE_1 : (batch_size_global, 27)
        view_directions = self.d_positional_encoding(view_directions)

        # LR_0 : (batch_size_global, 256)
        lr_0 = self.lr_zero(torch.cat([volume_features, colours], dim=1))

        # LR_1 : (batch_size_global, 256)
        values = self.lr_one(view_positions)

        # LR_2 - LR_4 : (batch_size_global, 256)
        for lr_i in self.lr_two_four:
            values = values * lr_0
            values = lr_i(values)

        # LR_5 - LR_7 : (batch_size_global, 256)
        values = self.lr_five(torch.cat([view_position_skip, values], -1))
        values = self.lr_six(values)
        values = self.lr_seven(values)

        # sigma : (batch_size_global, 1)
        sigma = self.density_out(values)

        # LR_8 : (batch_size_global, 256)
        values = self.lr_eight(values)
        values = self.lr_direction(torch.cat([view_directions, values], -1))

        # c : (batch_size_global, 3)
        colour = torch.sigmoid(self.colour_out(values))

        return colour, sigma
