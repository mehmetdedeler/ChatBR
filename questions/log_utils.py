import logging


def get_logger():

    logger_obj = logging.getLogger()
    logger_obj.setLevel(logging.INFO)
    # 创建控制台格式处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level=logging.INFO)
    console_fmt = logging.Formatter(fmt="%(message)s")
    console_handler.setFormatter(console_fmt)

    # 创建输出文件处理器
    file_handler = logging.FileHandler('log.log', 'a')
    file_handler.setLevel(logging.INFO)
    file_fmt = logging.Formatter("%(asctime)s - [%(funcName)s-->line:%(lineno)d] - %(levelname)s:%(message)s")
    file_handler.setFormatter(file_fmt)

    logger_obj.addHandler(console_handler)
    logger_obj.addHandler(file_handler)
    return logger_obj
