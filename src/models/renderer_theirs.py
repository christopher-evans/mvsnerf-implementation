import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.nerf.positional_encoder import PositionalEncoder


torch.autograd.set_detect_anomaly(True)

def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class Renderer_ours(nn.Module):
    def __init__(self, network_depth=7, hidden_layer_neurons=256, input_ch=3, input_ch_views=3, input_ch_feat=8 + 3 * 3, skips=[4]):
        """
        """
        super(Renderer_ours, self).__init__()

        self.x_positional_encoding = PositionalEncoder(input_ch, 10)
        self.d_positional_encoding = PositionalEncoder(input_ch_views, 4)
        input_ch = self.x_positional_encoding.num_encoded_dims
        input_ch_views = self.d_positional_encoding.num_encoded_dims

        self.network_depth = network_depth
        self.hidden_layer_neurons = hidden_layer_neurons

        self.skips = skips

        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, hidden_layer_neurons, bias=True)] + \
            [nn.Linear(hidden_layer_neurons, hidden_layer_neurons, bias=True) if i not in self.skips else nn.Linear(hidden_layer_neurons + input_ch, hidden_layer_neurons) for i in range(network_depth - 1)]
        )
        self.pts_bias = nn.Linear(input_ch_feat, hidden_layer_neurons)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + hidden_layer_neurons, hidden_layer_neurons // 2)])

        self.feature_linear = nn.Linear(hidden_layer_neurons, hidden_layer_neurons)
        self.density_out = nn.Linear(hidden_layer_neurons, 1)
        self.colour_out = nn.Linear(hidden_layer_neurons // 2, 3)

    def init_weights(self):
        self.pts_linears.apply(weights_init)
        self.views_linears.apply(weights_init)
        self.feature_linear.apply(weights_init)
        self.density_out.apply(weights_init)
        self.colour_out.apply(weights_init)

    def forward(self, view_positions, view_directions, volume_features, colours):
        view_positions = self.x_positional_encoding(view_positions)
        volume_features = torch.cat([volume_features, colours], dim=1)
        view_directions = self.d_positional_encoding(view_directions)

        h = view_positions
        bias = self.pts_bias(volume_features)
        for index, layer in enumerate(self.pts_linears):
            h = self.pts_linears[index](h) * bias
            h = F.relu(h)
            if index in self.skips:
                h = torch.cat([view_positions, h], -1)

        density_prediction = torch.relu(self.density_out(h))
        feature = self.feature_linear(h)
        h = torch.cat([feature, view_directions], -1)

        for index, layer in enumerate(self.views_linears):
            h = self.views_linears[index](h)
            h = F.relu(h)

        colour_prediction = torch.sigmoid(self.colour_out(h))

        return colour_prediction, density_prediction