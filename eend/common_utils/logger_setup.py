import logging
from collections import OrderedDict

# ------------------------------ Logger ------------------------------
# log to console or a file
def get_logger(
        name,
        format_str="%(asctime)s [%(pathname)s:%(lineno)s - %(levelname)s ] %(message)s",
        date_format="%Y-%m-%d %H:%M:%S",
        log_console=False):

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Create handlers
    f_handler = logging.FileHandler(name)
    f_handler.setLevel(logging.INFO)

    # Create formatters and add it to handlers
    f_format = logging.Formatter(fmt=format_str, datefmt=date_format)
    f_handler.setFormatter(f_format)

    # Add handlers to the logger
    logger.addHandler(f_handler)

    if log_console:
        c_handler = logging.StreamHandler()
        c_handler.setLevel(logging.INFO)
        c_format = logging.Formatter(fmt=format_str, datefmt=date_format)
        c_handler.setFormatter(c_format)
        logger.addHandler(c_handler)

    return logger


# set log info and format
def get_log_info(rank, epoch, iter_idx, metrics, lr, log_type='train'):
    log_key2format = OrderedDict({
        'loss': '{:.3f}',
        'loss_standard': '{:.3f}',
        'loss_attractor': '{:.3f}',
        'avg_ref_spk_qty': '{:.2f}',
        'avg_pred_spk_qty': '{:.2f}',
        'DER_miss': '{:.1f} %',
        'DER_FA': '{:.1f} %',
        'DER_conf': '{:.1f} %',
    })

    log_list = []
    for key in log_key2format:
        sub_log_str = ('{}: ' + log_key2format[key]).format(key, metrics[key])
        log_list.append(sub_log_str)
    log_metric_str = ' '.join(log_list)

    log_str = '[{}] Rank-{:03} Epoch-{:03}-Iter-{:05} '.format(log_type, rank, epoch, iter_idx) \
                            + ' '.join(log_list) \
                            + ' lr: {:.3e}'.format(lr)

    return log_str

