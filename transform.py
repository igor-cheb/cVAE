"""Base class for data transfomers."""
import abc
import typing

import torch


class Base(abc.ABC):
    """Base class for data transfomers."""

    @abc.abstractmethod
    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Transform the data."""

    def __init__(self, **_kwargs: typing.Any) -> None:  # noqa: B027
        """Initialise the transformer.

        This init is used as a default swallower of kwargs for transforms
        without inputs.
        """

    def inverse(self, data: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        """Inverse transformation.

        By default data is simply passed through. This is done to enable bulk
        inversion for all transforms for a given data module. If a transform
        does not have a defined inversion method, it will simply pass through
        the input data.
        """
        return data


class Pass(Base):
    """Transformer that does nothing."""

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Return the data without any changes."""
        return data


class ReverseTime(Base):
    """Transformer that flips time dimension of a tensor."""

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Return the data with reversed time dimension.

        This is to be used for recurrent encoders for the case of
        interpolation training.

        Args:
            data: Tensor containing the data,
            expected format is (Batch, Time, Features),
            where Time is sequence of steps in time.
        """
        return torch.flip(data, dims=[1])

    def inverse(self, data: torch.Tensor) -> torch.Tensor:
        """Inverse transformation."""
        return torch.flip(data, dims=[1])


class Quantise(Base):
    """Transform continuous tensor to positive discrete values."""

    def __init__(self, num_categories: int = 5, **_kwargs: typing.Any) -> None:
        """Initialise the transformer."""
        self.num_categories = num_categories

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Return quantised data."""
        min_negative_val = data.min() if data.min() < 0 else 0
        range_val = data.max() - data.min()

        return (
            (
                (data - min_negative_val) / range_val  # shift to positive
            )  # scale to [0, 1]
            * self.num_categories  # scale to [0, num_categories]
        ).round()  # round to nearest integer with .5 as threshold


class Normalise(Base):
    """Normalise data by standard deviation."""

    def __init__(
        self, mean: torch.Tensor, std: torch.Tensor, **_kwargs: typing.Any
    ) -> None:
        """Initialise the transformer."""
        self.mean = mean
        self.std = std

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Return normalised data."""
        return (data - self.mean.to(data.device)) / self.std.to(data.device)

    def inverse(self, data: torch.Tensor) -> torch.Tensor:
        """Inverse transformation."""
        return (data * self.std.to(data.device)) + self.mean.to(data.device)
