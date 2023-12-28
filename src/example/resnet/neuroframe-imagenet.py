import os, sys
import tqdm
import time

import torch
import torchvision
import numpy as np

from typing import Union, Callable

import sys, os
if os.environ.get("USE_LOCAL_DYNLIB", False):
    sys.path.append(os.environ.get("USE_LOCAL_DYNLIB"))
import neuroframe as nf

from lib import Timer

# Hyperparameters

IMAGENET_DATA_PATH = "/data/ImageNet_ILSVRC2012/imagenet"

model_configs = {
    'resnet18': {
        'train_batch_size': 384,
        'val_batch_size': 384,
        'init_lr': 1e-3,
    },
    'resnet34': {
        'train_batch_size': 256,
        'val_batch_size': 256,
        'init_lr': 1e-3,
    },
    'resnet152': {
        'train_batch_size': 32,
        'val_batch_size': 32,
        'init_lr': 1e-3,
    }
}

model = 'resnet34'

TRAIN_BATCH_SIZE = model_configs[model]['train_batch_size']
VAL_BATCH_SIZE = model_configs[model]['val_batch_size']

RUN_VAL_INTERVAL = 2700   # Run validation every RUN_VAL_INTERVAL batches
PRINT_PRED_RESULT_INTERVAL = 128

DEVICE = nf.Device.cuda(0)
DTYPE = nf.float32
NUM_EPOCHS = 90

LEARNING_RATE = model_configs[model]['init_lr']
LEARNING_RATE_DECAY = 0.1
LEARING_RATE_DECAY_INTERVAL = 30
# optimizer = nf.optim.SGD(0.9, 0.0001)
optimizer = nf.optim.Adam(0.9, 0.999, 1e-8)

class Scheduler:
    def __init__(self, init_lr: float, decay_rate: float, decay_interval: int):
        self.decay_rate = decay_rate
        self.decay_interval = decay_interval
        self.cur_lr = init_lr
        self.cur_step = 0
    
    def step(self) -> float:
        self.cur_step += 1
        if self.cur_step % self.decay_interval == 0:
            self.cur_lr *= self.decay_rate
        return self.cur_lr

def generate_random_tensor(shape: tuple[int, ...], dtype: nf.dtype) -> nf.Tensor:
    numel = np.prod(shape)
    bound = float(np.sqrt(1 / numel))
    return nf.Tensor.randu(shape, dtype, nf.Scalar(-bound), nf.Scalar(bound))

class Conv:
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        have_bias: bool = True,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.have_bias = have_bias
        
        kernel_weight = torch.nn.init.kaiming_normal_(torch.empty((out_channels, in_channels, kernel_size, kernel_size)), mode="fan_out", nonlinearity="relu")
        self.kernel = nf.misc.copy_cpu_torch_tensor_to_gpu_nf_tensor(kernel_weight.data_ptr(), kernel_weight.shape, DTYPE)
        # self.kernel = generate_random_tensor((out_channels, in_channels, kernel_size, kernel_size), DTYPE)
        optimizer.add_focus(self.kernel)
        
        if have_bias:
            self.bias = generate_random_tensor((out_channels, 1, 1), DTYPE)
            optimizer.add_focus(self.bias)
    
    def __call__(self, x: nf.Tensor) -> nf.Tensor:
        x = nf.ops.batched_convolution(x, self.kernel, self.stride, self.dilation)
        if self.have_bias:
            x = x + nf.ops.broadcast_to(self.bias, x.shape)
        return x


class BatchNorm:
    def __init__(
        self,
        num_channels: int,
        momentum: float = 0.9,
        eps: float = 1e-5
    ):
        self.num_channels = num_channels
        self.eps = eps
        self.momentum = momentum
        
        self.gamma = nf.Tensor.fill(nf.Scalar(1.0), (num_channels,), DTYPE)
        self.beta = nf.Tensor.zeros((num_channels,), DTYPE)
        # self.gamma = generate_random_tensor((num_channels,), DTYPE)
        # self.beta = generate_random_tensor((num_channels,), DTYPE)
        self.running_mean = nf.Tensor.zeros((num_channels,), DTYPE)
        self.running_var = nf.Tensor.fill(nf.Scalar(1.0), (num_channels,), DTYPE)
        
        optimizer.add_focus(self.gamma)
        optimizer.add_focus(self.beta)
    
    def __call__(self, x: nf.Tensor) -> nf.Tensor:
        return nf.ops.batch_norm(x, self.gamma, self.beta, self.running_mean, self.running_var, self.momentum, self.eps)


class MLP:
    def __init__(
        self,
        in_features: int,
        out_features: int,
        have_bias: bool = True
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.have_bias = have_bias
        
        bound = float(np.sqrt(1 / in_features))
        self.weight = nf.Tensor.randu((in_features, out_features),DTYPE, nf.Scalar(-bound), nf.Scalar(bound))
        # self.weight = generate_random_tensor((in_features, out_features), DTYPE)
        optimizer.add_focus(self.weight)
        
        if have_bias:
            self.bias = generate_random_tensor((out_features,), DTYPE)
            optimizer.add_focus(self.bias)
    
    def __call__(self, x: nf.Tensor) -> nf.Tensor:
        x = nf.ops.matmul(x, self.weight)
        if self.have_bias:
            x = x + nf.ops.broadcast_to(self.bias, x.shape)
        return x


class Pool:
    def __init__(
        self,
        pool_size: int
    ):
        self.pool_size = pool_size
    
    def __call__(self, x: nf.Tensor) -> nf.Tensor:
        return nf.ops.pool(x, self.pool_size)
    
    
class Relu:
    def __init__(self):
        pass
    
    def __call__(self, x: nf.Tensor) -> nf.Tensor:
        return nf.ops.relu(x)


class Sequential:
    def __init__(self, *layers):
        self.layers = layers
    
    def __call__(self, x: nf.Tensor) -> nf.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
    
    
class BasicBlock:
    expansion: int = 1
    
    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Sequential = None,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer = None
    ):
        if norm_layer is None:
            norm_layer = BatchNorm
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        if base_width != 64:
            raise NotImplementedError("Base width != 64 not supported in BasicBlock")

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = Conv(inplanes, planes, 3, stride, 1, False)
        self.bn1 = norm_layer(planes)
        self.relu = Relu()
        self.conv2 = Conv(planes, planes, 3, 1, 1, False)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
    
    def __call__(self, x: nf.Tensor) -> nf.Tensor:
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out = out + identity
        out = self.relu(out)
        
        return out


class Bottleneck:
    expansion: int = 4
    
    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Sequential = None,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer = None
    ) -> None:
        if norm_layer is None:
            norm_layer = BatchNorm
        width = int(planes * (base_width / 64.))
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = Conv(inplanes, width, 1, 1, 1, False)
        self.bn1 = norm_layer(width)
        self.conv2 = Conv(width, width, 3, stride, dilation, False)
        self.bn2 = norm_layer(width)
        self.conv3 = Conv(width, planes * self.expansion, 1, 1, 1, False)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = Relu()
        self.downsample = downsample
        self.stride = stride
    
    def __call__(self, x: nf.Tensor) -> nf.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        out = out + identity
        out = self.relu(out)
        
        return out
    
class ResNet:
    def __init__(
        self,
        block: type[Union[BasicBlock, Bottleneck]],
        layers: list[int],
        num_classes: int = 1000,
        width_per_group: int = 64,
        norm_layer = None
    ):
        if norm_layer is None:
            norm_layer = BatchNorm
        self._norm_layer = norm_layer
        
        self.inplanes = 64
        self.dilation = 1
        self.base_width = width_per_group
        self.conv1 = Conv(3, self.inplanes, 7, 2, 1, False)
        self.bn1 = norm_layer(self.inplanes)
        self.maxpool = Pool(2)
        self.relu = Relu()
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.fc = MLP(512 * block.expansion, num_classes)
    
    def _make_layer(
        self,
        block: type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1
    ) -> Sequential:
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = Sequential(
                Conv(self.inplanes, planes * block.expansion, 1, stride),
                norm_layer(planes * block.expansion)
            )
        
        layers = [
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.base_width,
                self.dilation,
                norm_layer
            )
        ]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer
                )
            )
        
        return Sequential(*layers)

    def forward(self, x: nf.Tensor) -> nf.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.maxpool(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # x: [bs, 512, 7, 7]
        x = nf.ops.reshape(x, (x.shape[0], x.shape[1], 7*7))
        # x: [bs, 512, 49]
        x = nf.ops.tensor_reduction_sum(x, 2)
        x = nf.ops.tensor_divs(x, 49)
        # x: [bs, 512]
        x = self.fc(x)
        
        return x

def get_resnet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

def get_resnet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])

def get_resnet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])

def get_resnet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])

def get_resnet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])
    
class ProactiveDataLoader:
    def __init__(
        self,
        dataset: torchvision.datasets.ImageFolder,
        dataloader: Callable[[torchvision.datasets.ImageFolder], torch.utils.data.DataLoader]
    ):
        self.dataset = dataset
        self.dataloader = dataloader(dataset)
        self.batch_size = self.dataloader.batch_size
        self.drop_last = self.dataloader.drop_last
        
    def fetch_next_batch(self):
        """
        This function fetches the next batch, and convert it to neuroframe tensor
        
        This function is blocking. It is designed to be called after the training
        thread has issued all GPU kernels and is waiting for the GPU to finish.
        """
        def fetch():
            images, target = next(self.loader_iter)
            target = target.to(torch.int32)
            assert images.dtype == torch.float32
            assert target.dtype == torch.int32
            if DTYPE == nf.float16:
                images = images.to(torch.float16)
            images = nf.misc.copy_cpu_torch_tensor_to_gpu_nf_tensor(images.data_ptr(), images.shape, DTYPE)
            target = nf.misc.copy_cpu_torch_tensor_to_gpu_nf_tensor(target.data_ptr(), target.shape, nf.int32)
            self.next_batch = images, target
            
        try:
            fetch()
        except StopIteration:
            self.next_batch = None
        
    def __iter__(self):
        self.loader_iter = iter(self.dataloader)
        self.next_batch = None
        self.fetch_next_batch()
        return self
    
    def __next__(self) -> tuple[nf.Tensor, nf.Tensor]:
        """
        Return the next batch
        """
        if self.next_batch is None:
            raise StopIteration
        return self.next_batch
            
    def get_data_count(self) -> int:
        if self.drop_last:
            return len(self.dataloader) * self.batch_size
        else:
            return len(self.dataset)
    
    def get_batch_count(self) -> int:
        return len(self.dataloader)
     
     
def run_validation(
    model: ResNet,
    dataloader: ProactiveDataLoader
):       
    with nf.inference_mode():
        val_set_batch_count = dataloader.get_batch_count()
        val_set_data_count = dataloader.get_data_count()
        val_correct_count = 0
        val_loss = nf.Tensor.zeros((), DTYPE)
        
        overall_timer = Timer()
        gpu_issue_timer = Timer()
        prefetch_timer = Timer()
        gpu_wait_timer = Timer()
        
        overall_timer.start()
        for i, (images, target) in enumerate(dataloader):
            gpu_issue_timer.start()
            output = model.forward(images)
            loss = nf.ops.tensor_reduction_sum(nf.ops.cross_entropy_loss(output, target))
            val_loss += loss
            gpu_issue_timer.end()
            
            prefetch_timer.start()
            dataloader.fetch_next_batch()
            prefetch_timer.end()
            
            gpu_wait_timer.start()
            val_correct_count += nf.ops.get_correct_sample_count(output, target)
            gpu_wait_timer.end()
            
            # print(f"Validation batch {i:4d}/{val_set_batch_count} ({i/val_set_batch_count*100:5.2f}%) | "
            #       f"Time usage: {gpu_issue_timer.get_time_usage_ms():4.2f} "
            #       f"{prefetch_timer.get_time_usage_ms():6.2f} "
            #       f"{gpu_wait_timer.get_time_usage_ms():6.2f} (ms)")
            
        overall_timer.end()
        val_loss = val_loss.numpy() / val_set_data_count
        val_correct_rate = val_correct_count / val_set_data_count
        print(f"Validation | Loss: {val_loss:10.8f} | Correct rate: {val_correct_rate*100:5.2f}% | "
              f"Time usage: {overall_timer.get_time_usage_ms()/1000:.2f} s")
        
if __name__ == "__main__":
    # ray.init()
    nf.Device.set_default_device(DEVICE)
    nf.set_random_seed(0)
    torch.manual_seed(0)
    
    normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    train_dataloader = ProactiveDataLoader(
        dataset = torchvision.datasets.ImageFolder(
            os.path.join(IMAGENET_DATA_PATH, 'train'),
            torchvision.transforms.Compose([
                torchvision.transforms.RandomResizedCrop(224),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                normalize,
            ])
        ),
        dataloader = lambda dataset: torch.utils.data.DataLoader(
            dataset,
            batch_size=TRAIN_BATCH_SIZE,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            prefetch_factor=4,
            drop_last=True,
            persistent_workers=True
        )
    )
    
    val_dataloader = ProactiveDataLoader(
        dataset = torchvision.datasets.ImageFolder(
            os.path.join(IMAGENET_DATA_PATH, 'val'),
            torchvision.transforms.Compose([
                torchvision.transforms.Resize(256),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                normalize,
            ])
        ),
        dataloader = lambda dataset: torch.utils.data.DataLoader(
            dataset,
            batch_size=VAL_BATCH_SIZE,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            drop_last=False
        )
    )
    
    model = {
        'resnet18': get_resnet18,
        'resnet34': get_resnet34,
        'resnet50': get_resnet50,
        'resnet101': get_resnet101,
        'resnet152': get_resnet152,
    }[model]()
    
    scheduler = Scheduler(LEARNING_RATE, LEARNING_RATE_DECAY, LEARING_RATE_DECAY_INTERVAL)
    
    learning_rate = LEARNING_RATE
    for epoch in tqdm.tqdm(range(NUM_EPOCHS)):
        print(f"---------------------------------")
        print(f"Epoch {epoch}. Start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}. Learning rate: {learning_rate}") 
        epoch_loss = nf.Tensor.zeros((), DTYPE)
        epoch_correct_count = 0
        
        train_set_batch_count = train_dataloader.get_batch_count()
        train_set_data_count = train_dataloader.get_data_count()
        
        batch_timer = Timer()
        gpu_issue_timer = Timer()
        prefetch_timer = Timer()
        gpu_wait_timer = Timer()
        
        for i, (images, target) in enumerate(train_dataloader):
            batch_timer.start()
            
            gpu_issue_timer.start()
            nf.cgraph.clear_graph()
            batch_size = images.shape[0]
            output = model.forward(images)
            loss = nf.ops.tensor_reduction_sum(nf.ops.cross_entropy_loss(output, target))
            
            with nf.inference_mode():
                epoch_loss += loss
                cur_mb_correct_count = nf.ops.get_correct_sample_count(output, target)
                epoch_correct_count += cur_mb_correct_count
                
                if i % PRINT_PRED_RESULT_INTERVAL == 0:
                    output_np = output.numpy()
                    pred = np.argmax(output_np, axis=1)
                    print(pred)
                
            loss = nf.ops.tensor_divs(loss, float(batch_size))
            
            nf.cgraph.perform_backward(loss, nf.Tensor.fill_like(nf.Scalar(1.0), loss))

            optimizer.step(learning_rate)
            gpu_issue_timer.end()
            
            prefetch_timer.start()
            # Here we've issued all kernels to GPU, so we load the next batch in advance
            train_dataloader.fetch_next_batch()
            prefetch_timer.end()
            
            gpu_wait_timer.start()
            loss_value = loss.numpy()
            gpu_wait_timer.end()
            
            batch_timer.end()
            
            print(f"Train batch {i:4d}/{train_set_batch_count} ({i/train_set_batch_count*100:5.2f}%) | "
                  f"Loss: {loss_value:10.8f} | "
                  f"Correct rate: {cur_mb_correct_count/batch_size*100:5.2f}% | "
                  f"Time usage: {batch_timer.get_time_usage_ms():4.2f}ms: "
                  f"{gpu_issue_timer.get_time_usage_ms():6.2f} "
                  f"{prefetch_timer.get_time_usage_ms():6.2f} "
                  f"{gpu_wait_timer.get_time_usage_ms():6.2f} (ms)")
            
            if i % RUN_VAL_INTERVAL == 0 and not (i == 0 and epoch == 0):
                print("Validation begin")
                nf.cgraph.clear_graph() # Release useless memory
                run_validation(model, val_dataloader)
                
            
        learning_rate = scheduler.step()
        epoch_loss = epoch_loss.numpy() / train_set_data_count
        epoch_correct_rate = epoch_correct_count / train_set_data_count * 100
        print(f"Epoch {epoch} | Loss: {epoch_loss: 2.7f} | Correct rate: {epoch_correct_rate: 4.2f}%")
        
            