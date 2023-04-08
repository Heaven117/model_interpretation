from typing import Callable, Tuple

import numpy as np
from skimage.segmentation import quickshift

from ianchor import Tasktype
from ianchor.candidate import AnchorCandidate


class ImageSampler():
    """
    Image sampling with the help of superpixels.
    The original input image is permuated by switching off superpixel areas.

    More details can be found on the following website:
    https://www.oreilly.com/content/introduction-to-local-interpretable-model-agnostic-explanations-lime/
    """

    type: Tasktype = Tasktype.IMAGE

    def __init__(self, input: any, predict_fn: Callable[[any], np.array], dataset: any = None):
        """
        Initialises ImageSampler with the given
        predict_fn, input and image dataset.

        Predict_fn will be used to predict all the
        samples and the input.

        When dataset equals None samples are generated
        by utilising mean superpixels.

        Args:
            input (any): Image that is to be explained.
            predict_fn (Callable[[any], np.array]): Black box model predict function.
            dataset (any): Image dataset from which samples will be collected
        """

        assert input.shape[2] == 3
        assert len(input.shape) == 3

        self.label = predict_fn(input[np.newaxis, ...])

        input = input.clone().cpu().detach().numpy()
        # run segmentation on the image
        self.features = quickshift(input.astype(np.double), kernel_size=4, max_dist=200, ratio=0.2)

        # parameters from original implementation
        segment_features = np.unique(self.features)
        self.num_features = len(segment_features)

        # create superpixel image by replacing superpixels by its mean in the original image
        self.sp_image = np.copy(input)
        for spixel in segment_features:
            self.sp_image[self.features == spixel, :] = np.mean(
                self.sp_image[self.features == spixel, :], axis=0
            )

        self.image = input
        self.predict_fn = predict_fn
        self.dataset = dataset

    def sample(
            self, candidate: AnchorCandidate, num_samples: int, calculate_labels: bool = True,
    ) -> Tuple[AnchorCandidate, np.ndarray]:
        """
        Generates num_samples samples by choosing random values
        out of self.dataset and setting the self.input features
        that are withing the candidates feature mask.

        When dataset is None then samples are generated by
        utilizing the mean superpixel else the image datset
        is sampled

        Args:
            candidate (AnchorCandidate): AnchorCandiate which contains the features to be fixated.
            num_samples (int): Number of samples that shall be generated.
            calculate_labels (bool, optional): When true label of the samples will predicted. In that case the
                candiates precision will be updated. Defaults to True.

        Returns:
            Tuple[AnchorCandidate, np.ndarray]: Structure: [AnchorCandiate, coverage_mask]. In case
            calculate_labels is False return [None, coverage_mask].
        """
        data = np.random.randint(
            0, 2, size=(num_samples, self.num_features)
        )  # generate random feature mask for each sample
        data[:, candidate.feature_mask] = 1  # set present features to one

        if not calculate_labels:
            return None, data

        # generate either samples from the dataset or mean superpixel
        if self.dataset is not None:
            return self.sample_dataset(candidate, data, num_samples)
        else:
            return self.sample_mean_superpixel(candidate, data, num_samples)

    def sample_dataset(
            self, candidate: AnchorCandidate, data: np.ndarray, num_samples: int,
    ) -> Tuple[AnchorCandidate, np.ndarray]:
        """
        Samples num_samples samples by utilising the image dataset.

        Args:
            candidate (AnchorCandidate): AnchorCandidate which precision will be updated.
            data (np.ndarray): Features masks
            num_samples (int): Number of samples to be generated.

        Returns:
            Tuple[AnchorCandidate, np.ndarray]: Structure: [AnchorCandiate, coverage_mask]
        """
        perturb_sample_idxs = np.random.choice(
            range(self.dataset.shape[0]), num_samples, replace=True
        )

        # generate samples from the dataset
        samples = np.stack(
            [
                self.__generate_image(mask, self.dataset[pidx])
                for mask, pidx in zip(data, perturb_sample_idxs)
            ],
            axis=0,
        )

        # predict samples
        preds = self.predict_fn(samples)
        labels = (preds == self.label).astype(int)

        # update candidate prec
        candidate.update_precision(np.sum(labels), num_samples)

        return candidate, data

    def sample_mean_superpixel(
            self, candidate: AnchorCandidate, data: np.ndarray, num_samples: int,
    ) -> Tuple[AnchorCandidate, np.ndarray]:
        """
        Sample function for image data.
        Generates random image samples from the distribution around the original image.

        Args:
            candidate (AnchorCandidate): AnchorCandidate which precision will be updated.
            data (np.ndarray): Generated feature mask.
            num_samples (int): Number of samples to be generated.

        Returns:
            Tuple[AnchorCandidate, np.ndarray]: Returns the AnchorCandiate and the feature masks.
        """
        # Sample function for image data.
        # Generates random image samples from the distribution around the original image.

        # Args:
        #     candidate (AnchorCandidate)
        #     num_samples (int)
        # Returns:
        #     candidate (AnchorCandidate)
        # """
        samples = np.stack([self.__generate_image(mask) for mask in data], axis=0)

        # predict labels
        preds = self.predict_fn(samples)
        labels = (preds == self.label).astype(int)

        # update candidate
        candidate.update_precision(np.sum(labels), num_samples)

        return candidate, data

    def __generate_image(self, feature_mask: np.ndarray) -> np.array:
        """
        Generate sample image given some feature mask.
        The true image will get permutated dependent on the feature mask.
        Pixel which are outmasked by the mask are replaced by the corresponding superpixel pixel.

        Args:
            feature_mask (np.ndarray): Feature mask to generate picture with

        Returns:
            np.array: Generated image.
        """
        img = self.image.copy()
        zeros = np.where(feature_mask == 0)[0]
        mask = np.zeros(self.features.shape).astype(bool)
        for z in zeros:
            mask[self.features == z] = True
        img[mask] = self.sp_image[mask]

        return img
