import logging, socket, os

# Log-Related Variables

def init_logger(args, rank, output_dir="./logs"):

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    extra = {
        "world_size": args.world_size,
        "lr": args.learning_rate,
        "dbs": "enabled" if args.dynamic_batch_size else "disabled",
        "ft": "enabled" if args.fault_tolerance else "disabled"
    }

    logger = logging.getLogger(socket.gethostname())
    for hdlr in logger.handlers[:]:
        logger.removeHandler(hdlr)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s [%(world_size)s:%(lr)s:dbs_%(dbs)s:ft_%(ft)s] [%(filename)s:%(lineno)d] %(levelname)s %(message)s')
    sh = logging.StreamHandler()
    sh.setLevel(logging.DEBUG)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    log_file = os.path.join(output_dir, '%s-debug_%d-n_%d-bs_%d-lr_%.4f-ep_%d-dbs_%d-ft_%d-ftc_%f-node%d-ocp%d.log'
                            % (args.model, int(args.debug), args.world_size, args.batch_size,
                               args.learning_rate, args.epoch_size, int(args.dynamic_batch_size),
                               int(args.fault_tolerance), args.fault_tolerance_chance,
                               rank, int(args.one_cycle_policy)))
    fh = logging.FileHandler(log_file, 'w+')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger = logging.LoggerAdapter(logger, extra)
    return logger