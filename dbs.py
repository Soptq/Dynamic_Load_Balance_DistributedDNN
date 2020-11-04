import random, math, json

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

import utils

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
dataset = args.dataset
dbs_enabled = args.dynamic_batch_size
gpu = args.gpu
training_model = args.model
ft_enabled = args.fault_tolerance
ftc = args.fault_tolerance_chance
ocp_enabled = args.one_cycle_policy
_disabled_enhancements = args.disable_enhancements

"""
##########################################################################################
#
#   Initialize Useful Variables
#
##########################################################################################
"""
# Saved file name
base_filename = '%s-%s-debug%d-n%d-bs%d-lr%.4f-ep%d-dbs%d-ft%d-ftc%f-node%s-ocp%d'\
                            % (args.model, args.dataset, int(args.debug), args.world_size, args.batch_size,
                               args.learning_rate, args.epoch_size, int(args.dynamic_batch_size),
                               int(args.fault_tolerance), args.fault_tolerance_chance,
                               "{}", int(args.one_cycle_policy))

if _disabled_enhancements:
    base_filename = "puredbs=" + base_filename

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
fault_wait = False  # Flag that indicates if current worker is in a random waiting phase.
fault_round = 0  # Random integer that indicates when will current worker stop waiting.
fault_wait_time = 0  # Random integer that indicates how many seconds current worker needs to wait.
current_epoch = -1  # A variable that stores current epoch number.

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
    global fault_round, fault_wait, ftc, ft_enabled, fault_wait_time, saved_epoch

    if not ft_enabled:
        return

    if fault_wait:  # Current worker is in a waiting phase
        if epoch <= fault_round:  # waiting is not completed, wait.
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
    return val_loss / num_batches, accuracy


def transformer_validate(val_loader, model, criterion, epoch, num_batches, ntokens, bptt):
    model.eval()
    # total = 0
    # correct = 0
    val_loss = 0
    with torch.no_grad():
        for i in range(0, val_loader.size(0) - 1, bptt):
            inputs, target = utils.get_batch(val_loader, i, bptt)
            inputs = inputs.to(DEVICE)
            target = target.to(DEVICE)
            output = model(inputs)
            output = output.view(-1, ntokens)
            val_loss += len(inputs) * criterion(output, target).item()

    val_loss /= (len(val_loader) - 1)
    logger.info(
        f'Rank {dist.get_rank()}, epoch {epoch}, val_loss {val_loss / num_batches}, accuracy {1 - val_loss}')
    return val_loss / num_batches, 1 - val_loss


"""
##########################################################################################
#
#   Model Training
#
##########################################################################################
"""


def adjust_learning_rate(optimizer, epoch):
    global lr, epoch_size
    """
    One Cycle Policy
    0 <= epoch < 0.3 * epoch_size: 0.01 * lr + ((0.99 * lr) / (epoch_size * 0.3)) * epoch
    0.3 * epoch_size <= epoch < 0.7 * epoch_size: lr
    0.7 * epoch_size <= epoch < epoch_size: lr - ((0.99 * lr) / (epoch_size * 0.3)) * (epoch - 0.7 * epoch)
    """

    if _disabled_enhancements:
        return


    # if 0 <= epoch < 0.3 * epoch_size:
    #     _lr = 0.01 * lr + ((0.99 * lr) / (0.3 * epoch_size)) * epoch
    # elif 0.7 * epoch_size:
    if 0.7 * epoch_size <= epoch < epoch_size:
        _lr = lr - ((0.99 * lr) / (0.3 * epoch_size)) * (epoch - 0.7 * epoch)
    else:
        _lr = lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = _lr


def train(trainloader, model, optimizer, criterion, epoch, num_batches, partition_size):
    _rank = dist.get_rank()
    _world_size = dist.get_world_size()
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
        wait_time = SSGD(model, _rank, _world_size, partition_size)  # Model averaging
        optimizer.step()
        epoch_loss += loss.item()
        running_loss += loss.item()
        train_time = time.time() - start_time
        average_time += wait_time
        if i % 10 == 0 and i > 0:
            logger.info(
                f'Rank {_rank}, epoch {epoch}: {i}, train_time {train_time}, average_time {average_time}, train_loss {running_loss / 10.0}')
            running_loss = 0.0
    train_time = time.time() - start_time
    logger.info(
        f'Rank {_rank}, epoch {epoch}, train_time {train_time}, train_loss {epoch_loss / num_batches}')
    return train_time - average_time, average_time, epoch_loss / num_batches


def transformer_train(trainloader, model, optimizer, criterion, epoch, num_batches, partition_size, ntokens, bptt):
    _rank = dist.get_rank()
    _world_size = dist.get_world_size()
    model.train()
    epoch_loss = 0.0
    running_loss = 0.0
    average_time = 0.0
    dist.barrier()
    start_time = time.time()

    for batch, i in enumerate(range(0, trainloader.size(0) - 1, bptt)):
        inputs, target = utils.get_batch(trainloader, i, bptt)
        inputs = inputs.to(DEVICE)
        target = target.to(DEVICE)
        optimizer.zero_grad()
        model.zero_grad()
        output = model(inputs)
        output = output.view(-1, ntokens)
        loss = criterion(output, target)
        loss.backward()
        fault_tolerance_wait(epoch, num_batches, dist.get_rank())  # Tolerance test
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        wait_time = SSGD(model, _rank, _world_size, partition_size)  # Model averaging
        optimizer.step()
        epoch_loss += loss.item()
        running_loss += loss.item()
        train_time = time.time() - start_time
        average_time += wait_time
        if i % 10 == 0 and i > 0:
            logger.info(
                f'Rank {_rank}, epoch {epoch}: {i}, train_time {train_time}, average_time {average_time}, train_loss {running_loss / 10.0}')
            running_loss = 0.0
    train_time = time.time() - start_time
    logger.info(
        f'Rank {_rank}, epoch {epoch}, train_time {train_time}, train_loss {epoch_loss / num_batches}')
    return train_time - average_time, average_time, epoch_loss / num_batches


def SSGD(model, _rank, _world_size, partition_size: np.ndarray):
    wait_time = 0.0
    weighted = (partition_size[_rank] / partition_size.sum()) if not _disabled_enhancements else (1 / _world_size)
    for param in model.parameters():
        sync_data = weighted * param.grad.data
        req = dist.all_reduce(sync_data, op=dist.ReduceOp.SUM, async_op=True)
        send_start = time.time()
        req.wait()
        wait_time += time.time() - send_start
        param.grad.data = sync_data
    return wait_time


"""
##########################################################################################
#
#   Distributed Simulating Code
#
##########################################################################################
"""


def run(rank, size, seed=1234):
    global lr, debug_mode_enabled, dbs_enabled

    if rank == 0:
        data_recorder = {"epoch": [],
                         "train_loss": [],
                         "train_time": [],
                         "sync_time": [],
                         "val_loss": [],
                         "accuracy": [],
                         "partition": [],
                         "node_time": [],
                         "wallclock_time": [],
                         }

    logger.info(f'Initiating Rank {rank}, World Size {size}')
    torch.manual_seed(seed)

    # Configure training model

    num_classes = 10
    if args.dataset == "cifar100":
        num_classes = 100

    ntokens = 33278
    emsize = 200
    nhead = 2
    nhid = 200
    nlayers = 2
    dropout = 0.2
    bptt = 35

    if args.model == "mnistnet":
        import Net.MnistNet
        model = Net.MnistNet.MnistNet()
    if args.model == "resnet":
        import Net.Resnet
        model = Net.Resnet.ResNet101(num_classes)
    if args.model == "densenet":
        import Net.Densenet
        model = Net.Densenet.DenseNet121(num_classes)
    if args.model == "googlenet":
        import Net.GoogleNet
        model = Net.GoogleNet.GoogLeNet(num_classes)
    if args.model == "regnet":
        import Net.RegNet
        model = Net.RegNet.RegNetY_400MF(num_classes)
    if args.model == "transformer":
        import Net.Transformer
        model = Net.Transformer.TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout)
    model = model.to(DEVICE)

    for name, param in model.named_parameters():
        dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
        param.data /= float(size)
        
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    if args.model == "transformer":
        criterion = F.nll_loss
    else:
        criterion = F.cross_entropy

    # Initialize default batch size distribution
    # At the beginning we assume all workers have the same performance.
    nodes_time = np.array([1.0 for _ in range(size)])  # Training time of workers
    partition_size = np.array([1.0 / size for _ in range(size)])  # Dataset partition ratio

    # Start training
    logger.info(f'Rank {rank} start training')
    total_train_time = 0  # Count total train time

    for epoch in range(epoch_size):
        if ocp_enabled:
            adjust_learning_rate(optimizer, epoch)
        if dbs_enabled:
            # Calculated dataset partition ratio based on workers' training time and last epoch's partition ratio.
            partition_size = get_size(nodes_time, partition_size)
            logger.info(f"Rank {rank}, adjusted partition size to {partition_size}")
        # Using calculated partition size to split dataset, getting train_set, val_set, as well as corresponding
        # batch size of current worker
        train_set, val_set, bsz = \
            dataloader.partition_dataset(dataset, partition_size, rank, batch_size, seed)

        if args.model == "transformer":
            num_batches = len(train_set)
        else:
            num_batches = math.ceil(len(train_set.dataset) / float(bsz))  # Calculate how many iterations in this epoch.

        logger.info(
            f"Rank {rank}, number of batches {num_batches}, batch size {bsz}, "
            f"length {bsz * num_batches}")

        epoch_start_time = time.time()
        # train() returned train_time excludes the communication time.
        if args.model == "transformer":
            train_time, sync_time, train_loss = transformer_train(train_set, model, optimizer, criterion, epoch,
                                                      num_batches, partition_size, ntokens, bptt)
        else:
            train_time, sync_time, train_loss = train(train_set, model, optimizer, criterion, epoch, num_batches,
                                                      partition_size)

        total_train_time += time.time() - epoch_start_time  # Get time that includes communication time.

        if args.model == "transformer":
            val_loss, accuracy = transformer_validate(val_set, model, F.cross_entropy, epoch, num_batches,
                                                      ntokens, bptt)
        else:
            val_loss, accuracy = validate(val_set, model, F.cross_entropy, epoch, num_batches)

        if dbs_enabled:
            # Exchange pure train time for dataset partition ratio calculating in the next epoch.
            nodes_time = time_allreduce(torch.tensor([train_time], dtype=torch.float32).cpu(), rank, size)
            logger.info(f"Rank {rank}, total time {nodes_time}")

        # record statistic data
        if rank == 0:
            data_recorder["epoch"].append(epoch)
            data_recorder["train_time"].append(train_time)
            data_recorder["sync_time"].append(sync_time)
            data_recorder["train_loss"].append(train_loss)
            data_recorder["val_loss"].append(val_loss)
            data_recorder["accuracy"].append(accuracy)
            data_recorder["partition"].append(partition_size)
            data_recorder["node_time"].append(nodes_time)
            data_recorder["wallclock_time"].append(total_train_time)

    if rank == 0:
        npy_filename = base_filename.format(str(rank)) + ".npy"
        np.save(os.path.join("./statis", npy_filename), data_recorder)

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

    logger = dbs_logging.init_logger(args, rank, base_filename)

    fn(rank, size)


if __name__ == "__main__":
    if os.path.isfile(os.path.join("./logs", base_filename.format("0") + ".log")):
        print("")
        print("===========================")
        print("Had finished this experiments, skipping...")
        print("===========================")
        print("")
        exit(0)

    time.sleep(3)
    processes = []
    for rank in range(world_size):
        p = Process(target=init_processes, args=(rank, world_size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
