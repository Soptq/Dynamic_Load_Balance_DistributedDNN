import random, math

import os
import time

import numpy as np

import torch

torch.multiprocessing.set_start_method('spawn', force=True)
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.multiprocessing import Process

import parser
import dataloader
import dbs_logging

args = parser.get_parser().parse_args()

"""
##########################################################################################
#
#   Get Arguments From Parser.
#
##########################################################################################
"""

debug_mode_enabled = args.debug
world_size = args.world_size
batch_size = args.batch_size
lr = args.learning_rate
epoch_size = args.epoch_size
dbs_enabled = args.dynamic_batch_size
gpu = args.gpu
training_model = args.model
ft_enable = args.fault_tolerance
ftc = args.fault_tolerance_chance

"""
##########################################################################################
#
#   Initialize Useful Variables
#
##########################################################################################
"""

# Configure Processing Unit
if debug_mode_enabled:
    DEVICE = "cpu"
elif isinstance(gpu, int):
    DEVICE = "cuda:{}".format(gpu)
    torch.cuda.set_device(gpu)
else:
    # Will configure it when the worker process is spawned.
    DEVICE = None

# Fault-Tolerance-Related Variables
fault_wait = False      # Flag that indicates if current worker is in a random waiting phase.
fault_round = 0         # Random integer that indicates when will current worker stop waiting.
fault_wait_time = 0     # Random integer that indicates how many seconds current worker needs to wait.
current_epoch = -1      # A variable that stores current epoch number.

# Log-Related Variables
logger = None


"""
##########################################################################################
#
#   Code For Fault Tolerance Test
#
#   This snippet of code will automatically decide whether current worker will be
#   slowed down.
#
##########################################################################################
"""


def fault_tolerance_wait(epoch, batch_num, rank):
    global fault_round, fault_wait, ftc, ft_enable, fault_wait_time, saved_epoch

    if not ft_enable:
        return

    if fault_wait:  # Current worker is in a waiting phase
        if epoch <= fault_round:    # waiting is not completed, wait.
            # Need to split the fault_wait_time into batch_num parts, as fault_wait_time is for a epoch not a iteration.
            time.sleep(float(fault_wait_time) / float(batch_num))
            return
        else:
            fault_wait = False

    # Current worker is not waiting.
    if saved_epoch != epoch:
        saved_epoch = epoch
    else:
        return  # A worker can only enter below code once a epoch.

    # fault_wait is false, should try worker's luck to see if he needs to wait.
    luck = random.random()
    logger.info(f"Rank {rank} got a luck of {luck}, limit is {ftc}")
    if luck < ftc:
        # Back luck!
        # generate a wait round and a wait time
        fault_wait_time = random.randint(5, 10)  # generate a wait time between 5 seconds to 10 seconds.
        fault_round = random.randint(4, 20)  # generate a wait round between 4 iterations to 20 iterations.
        fault_round += epoch  # wait until fault_round epoch.
        fault_wait = True  # start waiting on next iterations.
        logger.info(
            f"Rank {rank} starts to have a {fault_wait_time} seconds more waiting until epoch {fault_round} !")
        return
    else:
        # Lucky! there is no waiting.
        return


"""
##########################################################################################
#
#   Model Validation
#
##########################################################################################
"""


def validate(val_loader, model, criterion, epoch, num_batches):
    model.eval()
    total = 0
    correct = 0
    val_loss = 0
    with torch.no_grad():
        for data in val_loader:
            inputs, target = data
            inputs = inputs.to(DEVICE)
            target = target.to(DEVICE)
            output = model(inputs)
            val_loss += criterion(output, target).item()
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    val_loss /= total
    accuracy = 100 * correct / total
    logger.info(
        f'Rank {dist.get_rank()}, epoch {epoch}, val_loss {val_loss / num_batches}, accuracy {accuracy}')


"""
##########################################################################################
#
#   Model Training
#
##########################################################################################
"""


def train(trainloader, model, optimizer, criterion, epoch, num_batches):
    model.train()
    epoch_loss = 0.0
    running_loss = 0.0
    average_time = 0.0
    dist.barrier()
    start_time = time.time()

    for i, data in enumerate(trainloader, 0):
        inputs, target = data
        inputs = inputs.to(DEVICE)
        target = target.to(DEVICE)
        optimizer.zero_grad()
        output = model(inputs)
        loss = criterion(output, target)
        loss.backward()
        fault_tolerance_wait(epoch, num_batches, dist.get_rank())  # Tolerance test
        wait_time = SSGD(model)   # Model averaging
        optimizer.step()
        epoch_loss += loss.item()
        running_loss += loss.item()
        train_time = time.time() - start_time
        average_time += wait_time
        if i % 10 == 0:
            logger.info(
                f'Rank {dist.get_rank()}, epoch {epoch}: {i}, train_time {train_time}, average_time {average_time}, train_loss {running_loss / (10.0 if i is not 0 else 1.0)}')
            running_loss = 0.0
    train_time = time.time() - start_time
    logger.info(
        f'Rank {dist.get_rank()}, epoch {epoch}, train_time {train_time}, train_loss {epoch_loss / num_batches}')
    return train_time - average_time


def SSGD(model):
    wait_time = 0.0
    for param in model.parameters():
        req = dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM, async_op=True)
        send_start = time.time()
        req.wait()
        wait_time += time.time() - send_start
        param.grad.data /= float(world_size)
    return wait_time


"""
##########################################################################################
#
#   Distributed Simulating Code
#
##########################################################################################
"""


def run(rank, size):
    global lr, debug_mode_enabled, dbs_enabled
    logger.info(f'Initiating Rank {rank}, World Size {size}')
    torch.manual_seed(1234)

    # Configure training model
    if debug_mode_enabled:
        import Net.MnistNet
        model = Net.MnistNet.MnistNet()
    else:
        if args.model == "resnet":
            import Net.Resnet
            model = Net.Resnet.ResNet101()
        if args.model == "densenet":
            import Net.Densenet
            model = Net.Densenet.DenseNet121()
        if args.model == "googlenet":
            import Net.GoogleNet
            model = Net.GoogleNet.GoogLeNet()
        if args.model == "regnet":
            import Net.RegNet
            model = Net.RegNet.RegNetY_400MF()
    model = model.to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.5)

    # Initialize default batch size distribution
    # At the beginning we assume all workers have the same performance.
    nodes_time = np.array([1.0 for _ in range(size)])   # Training time of workers
    partition_size = np.array([1.0 / size for _ in range(size)])    # Dataset partition ratio

    # Start training
    logger.info(f'Rank {rank} start training')
    total_train_time = 0    # Count total train time

    for epoch in range(epoch_size):
        if dbs_enabled:
            # Calculated dataset partition ratio based on workers' training time and last epoch's partition ratio.
            partition_size = get_size(nodes_time, partition_size)
            logger.info(f"Rank {rank}, adjusted partition size to {partition_size}")

        # Using calculated partition size to split dataset, getting train_set, val_set, as well as corresponding
        # batch size of current worker
        train_set, val_set, bsz = dataloader.partition_dataset(partition_size, rank, debug_mode_enabled, batch_size)
        num_batches = math.ceil(len(train_set.dataset) / float(bsz))    # Calculate how many iterations in this epoch.
        logger.info(
            f"Rank {rank}, number of batches {num_batches}, batch size {train_set.batch_size}, "
            f"length {train_set.batch_size * num_batches}")

        epoch_start_time = time.time()
        # train() returned train_time excludes the communication time.
        train_time = train(train_set, model, optimizer, F.cross_entropy, epoch, num_batches)
        total_train_time += time.time() - epoch_start_time  # Get time that includes communication time.

        validate(val_set, model, F.cross_entropy, epoch, num_batches)

        if dbs_enabled:
            # Exchange pure train time for dataset partition ratio calculating in the next epoch.
            nodes_time = time_allreduce(torch.tensor([train_time], dtype=torch.float32).cpu(), rank, size)
            logger.info(f"Rank {rank}, total time {nodes_time}")

    logger.info(f'Rank {rank} Terminated')
    logger.info(f'Rank {rank} Total Time:')
    logger.info(total_train_time)


"""
##########################################################################################
#
#   DBS Algorithm
#
##########################################################################################
"""


def get_size(nodes_time: np.ndarray, partition_size: np.ndarray):
    _sum = 0.0
    for i in range(world_size):
        _sum += (partition_size[i] / nodes_time[i])
    cons_k = 1 / _sum  # get constant_k
    distribution_ratio = np.divide(partition_size * cons_k, nodes_time)
    # get the most accurate batch_size split
    norm_batch = distribution_ratio * batch_size / distribution_ratio.sum()
    floor_norm_batch = np.floor(norm_batch)
    floor_sum = int(floor_norm_batch.sum())
    ceil_counter = batch_size - floor_sum  # will pick top k to ceil
    idx_ceil = (norm_batch - floor_norm_batch).argsort()[-ceil_counter:]
    idx_round = np.argwhere(norm_batch - floor_norm_batch >= 0.5).reshape(-1)
    _, idx_inter, _ = np.intersect1d(idx_ceil, idx_round, return_indices=True)
    idx = idx_ceil[idx_inter]
    floor_norm_batch[idx] += 1
    norm = floor_norm_batch / floor_norm_batch.sum()

    return norm


def time_allreduce(send_buff, rank, size):
    recv_buff = send_buff.clone()

    left = ((rank - 1) + size) % size
    right = (rank + 1) % size

    result = [send_buff.item()]

    for i in range(size - 1):
        # Send send_buff
        send_req = dist.isend(send_buff, right)
        dist.recv(recv_buff, left)
        result.append(recv_buff.item())
        send_req.wait()
        send_buff = recv_buff.clone()

    for i in range(rank, size - 1):
        result.insert(0, result.pop())

    result.reverse()
    return result


"""
##########################################################################################
#
#   Distributed Simulating Code
#
##########################################################################################
"""


def init_processes(rank, size, fn, backend='gloo'):
    global DEVICE, logger
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)

    # Configuring multiple GPU
    if not debug_mode_enabled and isinstance(gpu, list):
        DEVICE = "cuda:{}".format(gpu[rank])
        torch.cuda.set_device(gpu[rank])

    logger = dbs_logging.init_logger(args, rank)

    fn(rank, size)


if __name__ == "__main__":
    processes = []
    for rank in range(world_size):
        p = Process(target=init_processes, args=(rank, world_size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
