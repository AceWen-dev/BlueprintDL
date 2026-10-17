import hashlib
import logging
import os


def _formatter():
    return logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s', datefmt='%H:%M:%S'
    )


def get_logger(work_dir=None, name='dlkit'):
    """获取日志器。

    - 每个 run 目录使用独立子 logger,日志不会串到其它 run;
    - 文件 handler 延迟打开:未真正写日志时不会占用 log.txt,便于清理;
    - 文件 handler 由 ``close_logger`` / ``Trainer.close`` 关闭(Windows 必需);
    - 关闭后子 logger 仍通过父 logger 输出到控制台。
    """

    root = logging.getLogger(name)
    root.setLevel(logging.INFO)
    root.propagate = False
    if not root.handlers:
        console = logging.StreamHandler()
        console.setFormatter(_formatter())
        root.addHandler(console)

    if not work_dir:
        return root

    absolute = os.path.abspath(str(work_dir))
    suffix = hashlib.md5(absolute.encode('utf-8')).hexdigest()[:10]
    logger = logging.getLogger('%s.%s' % (name, suffix))
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    logger.propagate = True  # 控制台输出交给父 logger
    file_handler = logging.FileHandler(
        os.path.join(absolute, 'log.txt'), encoding='utf-8', delay=True
    )
    file_handler.setFormatter(_formatter())
    logger.addHandler(file_handler)
    return logger


def close_logger(logger):
    """关闭并移除 logger 上的 handler(释放 log.txt 文件句柄)。"""

    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)


__all__ = ['close_logger', 'get_logger']
