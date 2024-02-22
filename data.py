"""Module with toy data modules."""

"""Base class for data modules."""

import abc
import typing

import pytorch_lightning as pl
import torch

from transform import Base as trans_Base, Normalise, Pass


class Base(pl.LightningDataModule, abc.ABC):
    """Base class for data modules."""

    @abc.abstractmethod
    def setup(self) -> None:
        """Create and transform the dataset."""
        raise NotImplementedError

    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        transforms: typing.Optional[tuple[type[trans_Base], ...]] = None,
        transform_params: typing.Optional[dict[str, typing.Any]] = None,
    ) -> None:
        """Initialize the data module."""
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        if transforms is None:
            transforms = (Pass,)
        self.transforms = transforms
        if transform_params is None:
            transform_params = {}
        self.transform_params = transform_params

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """Return the training dataloader."""
        return torch.utils.data.DataLoader(
            dataset=self.train_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """Return the validation dataloader."""
        return torch.utils.data.DataLoader(
            dataset=self.val_data,
            batch_size=len(self.val_data),
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(
        self,
    ) -> torch.utils.data.DataLoader:  # pragma: no cover
        """Return the test dataloader."""
        return torch.utils.data.DataLoader(
            dataset=self.val_data,
            batch_size=len(self.val_data),
            num_workers=self.num_workers,
            shuffle=False,
        )

    def _bimodal_feature(self, bimodal_func: typing.Callable) -> torch.Tensor:
        """Generate a feature with bimodal distribution.

        Args:
            bimodal_func: Function to be used to generate the feature.
            E.g. torch.cos, torch.sin, torch.tan, etc.
        """
        eps = torch.normal(
            mean=torch.zeros(self.data_size, self.seq_len),
            std=torch.tensor(self.noise_std).repeat(
                self.data_size, self.seq_len
            ),
        ).unsqueeze(-1)

        return (
            bimodal_func(torch.arange(self.data_size * self.seq_len))
            .view(-1, self.seq_len)
            .unsqueeze(-1)
        ) + eps

    def _split_data(
        self, data: torch.Tensor, train_ratio: float = 0.8
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Split the data into train and validation sets."""
        train_len = int(train_ratio * len(data))
        val_len = len(data) - train_len
        train, val = torch.utils.data.random_split(data, (train_len, val_len))

        # val and train are of type Subset, but torch.Tensor is needed
        # for further data transformation, hence the ouput
        return train.indices, val.indices 

    def _transform(
        self,
        train_data: torch.Tensor,
        val_data: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the list data transforms to passed data."""
        for transformation in self.transforms:
            initiated = transformation(**self.transform_params)  # type: ignore
            train_data = initiated(train_data)
            val_data = initiated(val_data)
        return train_data, val_data

    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        """Inverse all applied transforms."""
        for transformation in reversed(self.transforms):
            initiated = transformation(**self.transform_params)  # type: ignore
            data = initiated.inverse(data)
        return data


class Conditional(Base):
    """Data module with noisy cosine data with 2 features.

    To be used for testing generation with different distributions.

    !!! example
        datamodule = Cosine()

        get dataloaders for pytorch_lightning.Trainer:
            datamodule.setup()
            train_dataloader = datamodule.train_dataloader()
            val_dataloader = datamodule.val_dataloader()

        get actual train data:
            datamodule.train_dataloader().dataset
    """

    def __init__(  # noqa: PLR0913
        self,
        batch_size: int = 128,
        num_workers: int = 4,
        noise_std: float = 0.05,
        data_size: int = 5000,
        seq_len: int = 12,
        dependent_func: typing.Callable = torch.cos,
        add_condition: bool = True,  # noqa: FBT002, FBT001
        transforms: typing.Optional[tuple[type[trans_Base], ...]] = (
            Normalise,
        ),
        transform_params: typing.Optional[dict[str, typing.Any]] = None,
    ) -> None:
        """Initialize the data module.

        Args:
            batch_size: Batch size tb used in training.

            num_workers: Number of workers.

            noise_std: Standard deviation of noise to be added to the data.
                If 0, no noise is added.

            data_size: Batch dimension of the dataset tb generated.

            seq_len: Time dimension of the dataset,
                length of sequence in time.

            dependent_func: Function to be used to generate
                the dependent variable.

            add_condition: Whether to add a condition feature to the data.

            transforms: Tuple of data transformers.

            transform_params: Dictionary of parameters for data transformers.
        """
        super().__init__(
            batch_size=batch_size,
            num_workers=num_workers,
            transforms=transforms,
            transform_params=transform_params,
        )
        self.data_size = data_size
        self.seq_len = seq_len
        self.noise_std = noise_std
        self.add_condition = add_condition
        self.dependent_func = dependent_func

    def _noisy_timesteps(self, noise_std: float = 1.0) -> torch.Tensor:
        """Time steps with noise, sampled from normal distribtuion."""
        mean_tensor = (
            torch.arange(1, self.seq_len + 1).float().repeat(self.data_size, 1)
        )
        std_tensor = torch.tensor(noise_std).repeat(
            self.data_size, self.seq_len
        )
        # return absolute value to avoid negative time steps
        # and numerical problems with log function in dependent variable
        return torch.abs(torch.normal(mean=mean_tensor, std=std_tensor))

    def _condition(
        self, min_value: int = 1, max_value: int = 8
    ) -> torch.Tensor:
        """Variable representing the condition."""
        return torch.randint(
            low=int(min_value), high=int(max_value), size=(self.data_size, 1)
        ).repeat(1, self.seq_len)

    def _dependent(
        self,
        x: torch.Tensor,
        noise_std: float = 0.05,
    ) -> torch.Tensor:
        """Dependent variable y with noise eps."""
        eps = torch.normal(
            mean=torch.zeros(self.data_size, self.seq_len),
            std=torch.tensor(noise_std).repeat(self.data_size, self.seq_len),
        )
        self.condition = self._condition() if self.add_condition else 1
        return self.dependent_func(x * self.condition) + eps

    def setup(
        self,
        train_ratio: float = 0.8,
        stage: str | None = None,  # noqa: ARG002
    ) -> None:
        """Generate the dataset.

        Format of the generated data is (Batch, Time, Features).

        Args:
            train_ratio: Ratio of training data to total data.
                Remaining percentage is used for validation.
            stage: Stage of the experiment.
                Not used here, but required by pytorch lightning.
        """
        x = self._noisy_timesteps()
        y = self._dependent(x, noise_std=self.noise_std)

        stacked = torch.stack((x, y), dim=-1)
        # if self.add_condition:
        #     self.condition = self._condition()
            # stacked = torch.cat(
            #     (stacked, self.condition.unsqueeze(-1)), dim=-1
            # )

        train_ixs, val_ixs = self._split_data(
            stacked, train_ratio=train_ratio
        )
        self.train_data = stacked[train_ixs]
        self.val_data = stacked[val_ixs]
        if self.add_condition:
            self.condition = self.condition[:, 0]
            self.train_condition = self.condition[train_ixs]
            self.val_condition = self.condition[val_ixs]

        self.transform_params["mean"] = self.train_data.mean(dim=0)
        self.transform_params["std"] = self.train_data.std(dim=0)

        # flipping the time dimension and normalising the data
        self.train_data, self.val_data = self._transform(
            train_data=self.train_data, val_data=self.val_data
        )
        if self.add_condition:
            self.train_data = list(self._add_label(
                data=self.train_data, labels=self.train_condition
                )
            )
            self.val_data = list(self._add_label(
                data=self.val_data, labels=self.val_condition
                )
            )
    def _add_label(self, data: torch.Tensor, labels: torch.Tensor):
        for data_record, label in zip(data, labels):
            yield (data_record, label)
                