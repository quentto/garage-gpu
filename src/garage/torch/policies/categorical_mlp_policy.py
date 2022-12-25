"""CategoricalCNNPolicy."""
import akro
import torch
from torch import nn

from garage import InOutSpec
from garage.torch.modules import CNNModule, MultiHeadedMLPModule
from garage.torch.policies.stochastic_policy import StochasticPolicy


class CategoricalMLPPolicy(StochasticPolicy):
    """CategoricalMLPPolicy.

    A policy that contains a MLP to make prediction based on
    a categorical distribution.

    It only works with akro.Discrete action space.

    Args:
        env_spec (garage.EnvSpec): Environment specification.
        hidden_sizes (list[int]): Output dimension of dense layer(s) for
            the MLP for mean. For example, (32, 32) means the MLP consists
            of two hidden layers, each with 32 hidden units.
        hidden_nonlinearity (callable): Activation function for intermediate
            dense layer(s). It should return a torch.Tensor. Set it to
            None to maintain a linear activation.
        hidden_w_init (callable): Initializer function for the weight
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        hidden_b_init (callable): Initializer function for the bias
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        output_w_init (callable): Initializer function for the weight
            of output dense layer(s). The function should return a
            torch.Tensor.
        output_b_init (callable): Initializer function for the bias
            of output dense layer(s). The function should return a
            torch.Tensor.
        layer_normalization (bool): Bool for using layer normalization or not.
        name (str): Name of policy.

    """

    def __init__(
        self,
        env_spec,
        *,
        hidden_sizes=(32, 32),
        hidden_nonlinearity=torch.tanh,
        hidden_w_init=nn.init.xavier_uniform_,
        hidden_b_init=nn.init.zeros_,
        output_w_init=nn.init.xavier_uniform_,
        output_b_init=nn.init.zeros_,
        name="CategoricalMLPPolicy"
    ):

        if not isinstance(env_spec.action_space, akro.Discrete):
            raise ValueError(
                "CategoricalMLPPolicy only works " "with akro.Discrete action space."
            )
        if isinstance(env_spec.observation_space, akro.Dict):
            raise ValueError(
                "CNN policies do not support " "with akro.Dict observation spaces."
            )

        super().__init__(env_spec, name)

        self._mlp_module = MultiHeadedMLPModule(
            n_heads=1,
            input_dim=self._env_spec.observation_space.flat_dim,
            output_dims=[self._env_spec.action_space.flat_dim],
            hidden_sizes=hidden_sizes,
            hidden_w_init=hidden_w_init,
            hidden_b_init=hidden_b_init,
            hidden_nonlinearity=hidden_nonlinearity,
            output_w_inits=output_w_init,
            output_b_inits=output_b_init,
        )

    def forward(self, observations):
        """Compute the action distributions from the observations.

        Args:
            observations (torch.Tensor): Observations to act on.

        Returns:
            torch.distributions.Distribution: Batch distribution of actions.
            dict[str, torch.Tensor]: Additional agent_info, as torch Tensors.
                Do not need to be detached, and can be on any device.
        """
        mlp_output = self._mlp_module(observations)[0]
        logits = torch.softmax(mlp_output, axis=1)
        dist = torch.distributions.Categorical(logits=logits)
        return dist, {}
