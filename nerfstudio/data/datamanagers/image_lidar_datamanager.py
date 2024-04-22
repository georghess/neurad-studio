# Copyright 2024 the authors of NeuRAD and contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Lidar Datamanager.
"""

from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch
import torch.multiprocessing as mp
from rich.progress import Console, track
from typing_extensions import Literal

from nerfstudio.cameras.lidars import Lidars
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datamanagers.base_datamanager import TDataset
from nerfstudio.data.datamanagers.parallel_datamanager import ParallelDataManager, ParallelDataManagerConfig
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.datasets.lidar_dataset import LidarDataset
from nerfstudio.data.pixel_samplers import LidarPointSampler, LidarPointSamplerConfig, PixelSampler
from nerfstudio.data.utils.dataloaders import CacheDataloader, FixedIndicesEvalDataloader, RandIndicesEvalDataloader
from nerfstudio.data.utils.nerfstudio_collate import nerfstudio_collate
from nerfstudio.model_components.ray_generators import LidarRayGenerator, RayGenerator

CONSOLE = Console(width=120)


def lidar_variable_res_collate(batch: List[Dict]) -> Dict:
    """Default collate function for the cached dataloader.
    Args:
        batch: Batch of samples from the dataset.
    Returns:
        Collated batch.
    """
    lidars = []
    for data in batch:
        lidars.append(data.pop("lidar"))
    new_batch: dict = nerfstudio_collate(batch)
    new_batch["lidar"] = lidars
    return new_batch


def lidar_packed_collate(batch: List[Dict]) -> Dict:
    """Default collate function for the cached dataloader.
    Args:
        batch: Batch of samples from the dataset.
    Returns:
        Collated batch.
    """
    lidars = []
    for i, data in enumerate(batch):
        batch[i]["points_per_lidar"] = data["lidar"].shape[0]
        lidars.append(data.pop("lidar"))

    new_batch: dict = nerfstudio_collate(batch)
    new_batch["lidar"] = torch.cat(lidars, dim=0)
    return new_batch


@dataclass
class ImageLidarDataManagerConfig(ParallelDataManagerConfig):
    """A basic data manager"""

    _target: Type = field(default_factory=lambda: ImageLidarDataManager)
    """Target class to instantiate."""
    num_processes: int = 8
    """Number of processes to use for train data loading. More than 1 doesn't result in that much better performance"""
    queue_size: int = 8
    """Size of shared data queue containing generated ray bundles and batches."""
    train_num_lidar_rays_per_batch: int = 1024
    """Number of lidar rays per batch to use per training iteration."""
    eval_num_lidar_rays_per_batch: int = 1024
    """Number of lidar rays per batch to use per eval iteration."""
    downsample_factor: float = 1
    """Downsample factor for the lidar. If <1, downsample will be used."""
    max_thread_workers: Optional[int] = 1


class ImageLidarDataProcessor(mp.Process):  # type: ignore
    """Parallel dataset batch processor.

    This class is responsible for generating ray bundles from an input dataset
    in parallel python processes.

    Args:
        out_queue: the output queue for storing the processed data
        config: configuration object for the parallel data manager
        dataparser_outputs: outputs from the dataparser
        dataset: input dataset
        pixel_sampler: The pixel sampler for sampling rays
        ray_generator: The ray generator for generating rays
    """

    def __init__(
        self,
        out_queue: Any[mp.Queue],  # type: ignore
        func_queue: Any[mp.Queue],  # type: ignore
        config: ParallelDataManagerConfig,
        dataparser_outputs: DataparserOutputs,
        image_dataset: TDataset,
        pixel_sampler: PixelSampler,
        lidar_dataset: LidarDataset,
        point_sampler: LidarPointSampler,
        cached_images: Dict[str, torch.Tensor],
        cached_points: Dict[str, torch.Tensor],
    ):
        super().__init__()
        self.daemon = True
        self.out_queue = out_queue
        self.func_queue = func_queue
        self.config = config
        self.dataparser_outputs = dataparser_outputs
        self.image_dataset = image_dataset
        self.pixel_sampler = pixel_sampler
        self.lidar_dataset = lidar_dataset
        self.point_sampler = point_sampler
        self.ray_generator = RayGenerator(self.image_dataset.cameras)
        self.lidar_ray_generator = LidarRayGenerator(self.lidar_dataset.lidars)
        self.cached_images = cached_images
        self.cached_points = cached_points

    def run(self):
        """Append out queue in parallel with ray bundles and batches."""
        while True:
            while not self.func_queue.empty():
                func, args, kwargs = self.func_queue.get()
                func(self, *args, **kwargs)
            ray_bundle, batch = self.get_batch_and_ray_bundle()
            if torch.cuda.is_available():
                ray_bundle = ray_bundle.pin_memory()
            self.out_queue.put((ray_bundle, batch))

    def get_batch_and_ray_bundle(self):
        img_batch, img_ray_bundle = self.get_image_batch_and_ray_bundle()
        lidar_batch, lidar_ray_bundle = self.get_lidar_batch_and_ray_bundle()
        return _merge_img_lidar(img_ray_bundle, img_batch, lidar_ray_bundle, lidar_batch, len(self.image_dataset))

    def get_image_batch_and_ray_bundle(self):
        if not len(self.image_dataset.cameras):
            return None, None
        batch = self.pixel_sampler.sample(self.cached_images)
        ray_indices = batch["indices"]
        ray_bundle: RayBundle = self.ray_generator(ray_indices)
        return batch, ray_bundle

    def get_lidar_batch_and_ray_bundle(self):
        if not len(self.lidar_dataset.lidars):
            return None, None
        batch = self.point_sampler.sample(self.cached_points)
        ray_indices = batch.pop("indices")
        ray_bundle: RayBundle = self.lidar_ray_generator(ray_indices, points=batch["lidar"])
        return batch, ray_bundle


class ImageLidarDataManager(ParallelDataManager):
    """This extends the VanillaDataManager to support lidar data.

    Args:
        config: the DataManagerConfig used to instantiate class
    """

    config: ImageLidarDataManagerConfig
    train_lidar_dataset: LidarDataset
    eval_lidar_dataset: LidarDataset
    train_point_sampler: Optional[LidarPointSampler] = None
    eval_point_sampler: Optional[LidarPointSampler] = None

    def __init__(
        self,
        config: ImageLidarDataManagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,  # pylint: disable=unused-argument
    ):
        self.use_mp = config.num_processes > 0
        super().__init__(config, device, test_mode, world_size, local_rank, **kwargs)
        self.next_batch = None  # for prefetching to gpu in parallel

    def create_train_dataset(self) -> InputDataset:
        """Sets up the data loaders for training"""
        self.train_lidar_dataset = LidarDataset(
            dataparser_outputs=self.train_dataparser_outputs,
            downsample_factor=self.config.downsample_factor,
        )
        img_dataset = super().create_train_dataset()
        return img_dataset

    def create_eval_dataset(self) -> InputDataset:
        """Sets up the data loaders for evaluation"""
        self.eval_lidar_dataset = LidarDataset(
            dataparser_outputs=self.eval_dataparser_outputs,
            downsample_factor=self.config.downsample_factor,
        )
        img_dataset = super().create_eval_dataset()
        return img_dataset

    def setup_train(self):
        """Sets up the data loaders for training"""
        assert self.train_dataset is not None
        self.train_pixel_sampler = self._get_pixel_sampler(self.train_dataset, self.config.train_num_rays_per_batch)  # type: ignore
        self.train_point_sampler = LidarPointSamplerConfig().setup(
            num_rays_per_batch=self.config.train_num_lidar_rays_per_batch
        )
        self.train_lidar_ray_generator = LidarRayGenerator(
            self.train_lidar_dataset.lidars,
        )
        # Cache jointly to allow memory sharing between processes
        cached_images = _cache_images(self.train_dataset, self.config.max_thread_workers, self.config.collate_fn)
        cached_points = _cache_points(self.train_lidar_dataset, self.config.max_thread_workers, lidar_packed_collate)
        self.data_queue = mp.Queue(maxsize=self.config.queue_size) if self.use_mp else None
        # Create an individual queue for passing functions to each process
        self.func_queues = [mp.Queue() for _ in range(max(self.config.num_processes, 1))]
        self.data_procs = [
            ImageLidarDataProcessor(
                out_queue=self.data_queue,
                func_queue=func_queue,
                config=self.config,
                dataparser_outputs=self.train_dataparser_outputs,
                image_dataset=self.train_dataset,
                pixel_sampler=self.train_pixel_sampler,
                lidar_dataset=self.train_lidar_dataset,
                point_sampler=self.train_point_sampler,
                cached_images=cached_images,
                cached_points=cached_points,
            )
            for func_queue in self.func_queues
        ]
        if self.use_mp:
            for proc in self.data_procs:
                proc.start()
            print("Started processes")

    def setup_eval(self):
        """Sets up the data loader for evaluation"""
        if len(self.eval_dataset.cameras):
            super().setup_eval()
        else:
            CONSOLE.print("Setting up lidar-only eval dataset...")
            self.eval_camera_optimizer = None
        if not len(self.eval_lidar_dataset.lidars):
            CONSOLE.print("Not setting up lidar eval dataset...")
            self.eval_lidar_optimizer = None
            return
        self.eval_lidar_dataloader = CacheDataloader(
            self.eval_lidar_dataset,
            num_images_to_sample_from=self.config.eval_num_images_to_sample_from,
            num_times_to_repeat_images=self.config.eval_num_times_to_repeat_images,
            device=self.device,
            num_workers=self.world_size * 4,
            pin_memory=True,
            collate_fn=lidar_packed_collate,
        )
        self.iter_eval_lidar_dataloader = iter(self.eval_lidar_dataloader)
        self.eval_point_sampler = LidarPointSamplerConfig().setup(
            num_rays_per_batch=self.config.eval_num_lidar_rays_per_batch
        )
        self.eval_lidar_ray_generator = LidarRayGenerator(
            self.eval_lidar_dataset.lidars.to(self.device),
        )
        # for loading full images
        self.fixed_indices_eval_lidar_dataloader = FixedIndicesEvalDataloader(
            dataset=self.eval_lidar_dataset,
            device=self.device,
            num_workers=self.world_size * 4,
        )
        self.rand_indices_eval_lidar_dataloader = RandIndicesEvalDataloader(
            dataset=self.eval_lidar_dataset,
            device=self.device,
            num_workers=self.world_size * 4,
        )

    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the train dataloader."""
        self.train_count += 1
        ray_bundle, batch = self.next_batch if self.next_batch is not None else self._get_from_queue()
        if self.use_mp:
            self.next_batch = self._get_from_queue()  # prefetch next batch
        return ray_bundle, batch

    def _get_from_queue(self) -> Tuple[RayBundle, Dict]:
        if self.use_mp:
            ray_bundle, batch = self.data_queue.get()
        else:
            # Manually call the data processing function (not in parallel)
            ray_bundle, batch = self.data_procs[0].get_batch_and_ray_bundle()
        ray_bundle = ray_bundle.to(self.device, non_blocking=self.use_mp)
        batch = {k: v.to(self.device, non_blocking=self.use_mp) for k, v in batch.items()}
        return ray_bundle, batch

    def next_eval(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the eval dataloader."""
        if len(self.eval_dataset.cameras):
            img_ray_bundle, img_batch = super().next_eval(step)
        else:
            self.eval_count += 1
            img_ray_bundle, img_batch = None, None
        if len(self.eval_lidar_dataset.lidars):
            lidar_batch = next(self.iter_eval_lidar_dataloader)
            lidar_batch = self.eval_point_sampler.sample(lidar_batch)
            lidar_ray_bundle = self.eval_lidar_ray_generator(lidar_batch.pop("indices"), points=lidar_batch["lidar"])
        else:
            lidar_ray_bundle, lidar_batch = None, None
        return _merge_img_lidar(
            img_ray_bundle, img_batch, lidar_ray_bundle, lidar_batch, len(self.eval_dataset.cameras)
        )

    def get_num_train_data(self) -> int:
        """Get the number of training datapoints (images + lidar scans)."""
        return len(self.train_dataset.cameras) + len(self.train_lidar_dataset.lidars)

    def next_eval_lidar(self, step: int) -> Tuple[Lidars, Dict]:
        for lidar, batch in self.rand_indices_eval_lidar_dataloader:
            assert lidar.shape[0] == 1
            assert isinstance(lidar, Lidars)
            return lidar, batch
        raise ValueError("No more eval images")

    def clear_data_queue(self):
        """Clear everything in the queue + any potential batch in progress."""
        self.next_batch = None
        if self.data_queue is None:
            return
        for _ in range(self.config.queue_size + 2 * self.config.num_processes):
            self.data_queue.get()

    def __del__(self):
        """Close the data queue when the object is deleted."""
        if self.use_mp:
            super().__del__()


def _cache_images(image_dataset, workers, collate_fn):
    """Caches all input images into a NxHxWx3 tensor."""
    indices = range(len(image_dataset))
    batch_list = []
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        for idx in indices:
            res = executor.submit(image_dataset.__getitem__, idx)
            results.append(res)
        for res in track(results, description="Loading data batch", transient=False):
            batch_list.append(res.result())
    return collate_fn(batch_list)


def _cache_points(lidar_dataset, workers, collate_fn):
    """Caches all input images into a NxHxWx3 tensor."""
    indices = range(len(lidar_dataset))
    batch_list = []
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        for idx in indices:
            res = executor.submit(lidar_dataset.__getitem__, idx)
            results.append(res)
        for res in track(results, description="Loading data batch", transient=False):
            batch_list.append(res.result())
    return collate_fn(batch_list)


def _merge_img_lidar(
    img_ray_bundle: Optional[RayBundle],
    img_batch: Optional[Dict],
    lidar_ray_bundle: Optional[RayBundle],
    lidar_batch: Optional[Dict],
    img_dataset_len: int,
) -> Tuple[RayBundle, Dict]:
    """Helper function for merging img and lidar data."""
    if img_ray_bundle is None and lidar_ray_bundle is None:
        raise ValueError("Need either img or lidar data (or both)")

    # process image
    if img_ray_bundle is not None:
        assert img_batch is not None
        device = img_ray_bundle.origins.device
        img_batch["is_lidar"] = torch.zeros((len(img_ray_bundle), 1), dtype=torch.bool, device=device)
        img_batch["did_return"] = torch.ones((len(img_ray_bundle), 1), dtype=torch.bool, device=device)
        img_ray_bundle.metadata["is_lidar"] = img_batch["is_lidar"]
        img_batch["img_indices"] = img_batch.pop("indices")
        img_ray_bundle.metadata["did_return"] = torch.ones((len(img_ray_bundle), 1), dtype=torch.bool, device=device)

    # process lidar
    if lidar_ray_bundle is not None:
        assert lidar_batch is not None
        lidar_ray_bundle.camera_indices = lidar_ray_bundle.camera_indices + img_dataset_len
        lidar_batch["is_lidar"] = lidar_ray_bundle.metadata["is_lidar"]
        lidar_batch["distance"] = lidar_ray_bundle.metadata["directions_norm"]
        lidar_batch["did_return"] = lidar_ray_bundle.metadata["did_return"]

    # merge
    if img_ray_bundle is None or img_batch is None:
        ray_bundle, batch = lidar_ray_bundle, lidar_batch
    elif lidar_ray_bundle is None or lidar_batch is None:
        ray_bundle, batch = img_ray_bundle, img_batch
    else:
        ray_bundle = img_ray_bundle.cat(lidar_ray_bundle, dim=0)
        overlapping_keys = set(img_batch.keys()).intersection(set(lidar_batch.keys())) - {"is_lidar", "did_return"}
        assert not overlapping_keys, f"Overlapping keys in batch: {overlapping_keys}"
        batch = {
            **img_batch,
            **lidar_batch,
            "is_lidar": torch.cat([img_batch["is_lidar"], lidar_batch["is_lidar"]]),
            "did_return": torch.cat([img_batch["did_return"], lidar_batch["did_return"]]),
        }
    return ray_bundle, batch
