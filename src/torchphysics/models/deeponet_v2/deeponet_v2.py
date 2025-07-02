import torch
import torch.nn as nn


class TrunkNetV2(nn.Module):
    def __init__(self, model: nn.Module, input_dim: int, output_dim: int,
                 num_locations: int | None = None, num_channels: int | None = None,
                 add_output_layer: bool = False, model_output_size: int | None = None):
        """A trunk net that can be used in a DeepONet.
        
        Parameters
        ----------
        model : nn.Module
            The model that will be used as the trunk net.
        input_dim : int
            The dimension of the input space of the trunk net.
        output_dim : int
            The dimension of the output space.
        num_locations : int
            The number of num_locations in the trunk net. Only used if add_output_layer
            is True.
        num_channels : int
            The number of num_channels in the output layer. Refers to the number of
            $p$, the number of output neurons per location used in inner product.
            Only used if add_output_layer is True.
        add_output_layer : bool, optional
            If True, an output layer will be added to the trunk net. This is useful
            if the trunk net has a flat output (batch_size, model_output_size) and
            we want to reshape it to (batch_size, num_locations, output_dim, num_channels).
            Default is False.
        model_output_size : int, optional
            The size of the output of the model. If not None, the output layer will be
            initialized with this size. If None, the output layer will be inferred from
            a dummy input.
        """
        super(TrunkNetV2, self).__init__()
        self.model = model
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_locations = num_locations
        self.num_channels = num_channels
        self.add_output_layer = add_output_layer
        self.model_output_size = model_output_size
        self.output_layer = None
        if add_output_layer:
            self.output_layer = self._infer_shape_and_add_layer()
    
    def _infer_shape_and_add_layer(self):
        """Adds an output layer to the trunk net, if trunk net has flat output
        (batch_size, model_output_size) -> (batch_size, num_locations, output_dim * num_channels).
        """
        if self.model_output_size is not None:
            return nn.Linear(self.model_output_size, self.output_dim * self.num_channels)
        
        # infer the output shape from a dummy input
        dummy_input = torch.zeros((1, self.num_locations, self.input_dim))
        with torch.no_grad():
            dummy_output = self.model(dummy_input)
        if len(dummy_output.shape) != 3 and dummy_output.shape[:2] != (1, self.num_locations):
            raise ValueError(f"Output shape {dummy_output.shape} does not match expected "
                             f"shape (batch_size=1, model_output_size).")
        return nn.Linear(dummy_output.shape[-1], self.output_dim * self.num_channels)
    
    def _assert_input_shape(self, input_shape: torch.Tensor):
        """Asserts that the input shape is correct for the trunk net."""
        if len(input_shape.shape) != 3 or input_shape.shape[2] != self.input_dim:
            raise ValueError(f"Input shape {input_shape.shape} does not match expected "
                             f"shape (batch_size, num_locations, input_dim_trunk={self.input_dim}).")
        
    def _assert_output_shape(self, output_shape: torch.Tensor):
        """Asserts that the output shape is correct for the trunk net."""
        if len(output_shape.shape) != 4 or output_shape.shape[2] != self.output_dim:
            raise ValueError(f"Output shape {output_shape.shape} does not match expected "
                             f"shape (batch_size, num_locations, output_dim_trunk={self.output_dim}, "
                              "num_channels).")
    
    def reshape_output(self, output: torch.Tensor) -> torch.Tensor:
        """Reshapes the output of the trunk net to match the output space, num_locations
        and batch size. Reshapes from (batch_size, num_locations, output_dim * num_channels)
        to (batch_size, num_locations, output_dim, num_channels).

        Parameters
        ----------
        output : torch.Tensor
            The output of the trunk net.
        Returns
        -------
        torch.Tensor
            The reshaped output of the trunk net.
        """
        return output.reshape(output.shape[0], output.shape[1], self.output_dim,
                              self.num_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the trunk net to the given inputs.
        
        Parameters
        ----------
        x : torch.Tensor
            The inputs for the trunk net. Shape should be (batch_size, locations, input_dim).
        
        Returns
        -------
        torch.Tensor
            The output of the trunk net. Shape will be (batch_size, locations, output_dim, num_channels).
        """
        self._assert_input_shape(x)
        output = self.model(x)
        if self.output_layer is not None:
            output = self.output_layer(output)
            output = self.reshape_output(output)
        self._assert_output_shape(output)
        return output


class BranchNetV2(nn.Module):
    def __init__(self, model: nn.Module, input_dim: int, output_dim: int,
                 num_channels: int | None = None, num_sensors: int | None = None,
                 add_output_layer: bool = False, model_output_size: int | None = None):
        """A branch net that can be used in a DeepONet.
        
        Parameters
        ----------
        model : nn.Module
            The model that will be used as the branch net.
        input_dim : int
            The dimension of the input space of the branch net.
        output_dim : int
            The dimension of the output space.
        num_channels : int, optional
            The number of channels in the output layer. Refers to the number of
            $p$, the number of output neurons per location used in inner product.
            Only used if add_output_layer is True.
        add_output_layer : bool, optional
            If True, an output layer will be added to the branch net. This is useful
            if the branch net has a flat output (batch_size, model_output_size) and
            we want to reshape it to (batch_size, length, output_dim, num_channels).
            Default is False.
        model_output_size : int, optional
            The size of the output of the model. If not None, the output layer will be
            initialized with this size. If None, the output layer will be inferred from
            a dummy input.
        """
        super(BranchNetV2, self).__init__()
        self.model = model
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_channels = num_channels
        self.num_sensors = num_sensors
        self.add_output_layer = add_output_layer
        self.model_output_size = model_output_size
        self.output_layer = None
        if add_output_layer:
            self.output_layer = self._infer_shape_and_add_layer()

    def _infer_shape_and_add_layer(self):
        """Adds an output layer to the branch net, if branch net has flat output
        (batch_size, model_output_size) -> (batch_size, length, output_dim * num_channels).

        Returns
        -------
        nn.Module
            A linear layer (batch_size, output_dim * num_channels).
        """
        if self.model_output_size is not None:
            return nn.Linear(self.model_output_size, self.output_dim * self.num_channels)
        
        # infer the output shape from a dummy input
        dummy_input = torch.zeros((1, self.num_sensors, self.input_dim))
        with torch.no_grad():
            dummy_output = self.model(dummy_input)
        if len(dummy_output.shape) != 2 and dummy_output.shape[0] != 1:
            raise ValueError(f"Output shape {dummy_output.shape} does not match expected "
                             f"shape (batch_size=1, model_output_size).")
        return nn.Linear(dummy_output.shape[1], self.output_dim * self.num_channels)
    
    def _assert_input_shape(self, input_shape: torch.Tensor):
        """Asserts that the input shape is correct for the branch net."""
        if len(input_shape.shape) != 3 or input_shape.shape[2] != self.input_dim:
            raise ValueError(f"Input shape {input_shape.shape} does not match expected "
                             f"shape (batch_size, length, input_dim_branch={self.input_dim}).")
    
    def _assert_output_shape(self, output_shape: torch.Tensor):
        """Asserts that the output shape is correct for the branch net."""
        if len(output_shape.shape) != 3 or output_shape.shape[1] != self.output_dim:
            raise ValueError(f"Output shape {output_shape.shape} does not match expected "
                             f"shape (batch_size, output_dim_branch={self.output_dim}, "
                             "num_channels).")
        
    def reshape_output(self, output: torch.Tensor) -> torch.Tensor:
        """Reshapes the output of the branch net to match the output space, length
        and batch size. Reshapes from (batch_size, length, output_dim * num_channels)
        to (batch_size, length, output_dim, num_channels).

        Parameters
        ----------
        output : torch.Tensor
            The output of the branch net.
        
        Returns
        -------
        torch.Tensor
            The reshaped output of the branch net.
        """
        return output.reshape(output.shape[0], self.output_dim,
                              self.num_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the branch net to the given inputs.
        
        Parameters
        ----------
        x : torch.Tensor
            The inputs for the branch net. Shape should be (batch_size, length, input_dim).
        
        Returns
        -------
        torch.Tensor
            The output of the branch net. Shape will be (batch_size, output_dim, num_channels).
        """
        self._assert_input_shape(x)
        output = self.model(x)
        if self.output_layer is not None:
            output = self.output_layer(output)
            output = self.reshape_output(output)
        self._assert_output_shape(output)
        return output


class DeepONetV2(nn.Module):
    def __init__(self, trunk_net: TrunkNetV2, branch_net: BranchNetV2):
        super(DeepONetV2, self).__init__()
        self.trunk = trunk_net
        self.branch = branch_net
        self._verify_init()
    
    def forward(self, x_trunk, x_branch) -> torch.Tensor:
        trunk_output = self.trunk(x_trunk)
        branch_output = self.branch(x_branch)
        # (batch, output_dim, channels) * (batch, num_locations, output_dim, channels)
        # -> (batch, num_locations, output_dim)
        return torch.einsum('bod,bnod->bno', branch_output, trunk_output)

    def _verify_init(self):
        """Verifies that the trunk and branch nets are initialized correctly."""
        assert isinstance(self.trunk, TrunkNetV2), "trunk_net must be an instance of TrunkNet"
        assert isinstance(self.branch, BranchNetV2), "branch_net must be an instance of BranchNet"

        if self.trunk.output_dim != self.branch.output_dim:
            raise ValueError("The output dimensions of the trunk and branch nets must match.")
        
        if self.trunk.num_channels != self.branch.num_channels and \
            self.trunk.num_channels is not None and \
                self.branch.num_channels is not None:
            raise ValueError("The number of channels in the trunk and branch nets must match.")


