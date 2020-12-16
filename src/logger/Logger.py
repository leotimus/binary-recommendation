import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


def getLogger(name, logPath='logs/main.log', console=False) -> logging.Logger:
  """
  Simple logging wrapper that returns logger
  configured to log into file and console.

  Args:
      name (str): name of logger
      logPath (str): path of log file
      console (bool): whether to log on console

  Returns:
      logging.Logger: configured logger
  """
  name = Path(sys.argv[0]).name if name == "__main__" else name
  logger = logging.getLogger(name)
  logger.setLevel(logging.DEBUG)
  formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

  # ensure that logging handlers are not duplicated
  for handler in list(logger.handlers):
    logger.removeHandler(handler)

  # rotating file handler
  if logPath:
    Path(logPath).parent.mkdir(parents=True, exist_ok=True)
    fh = RotatingFileHandler(logPath,
                             maxBytes=10 * 2 ** 20,  # 10 MB
                             backupCount=1)  # 1 backup
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

  # console handler
  if console:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

  # null handler
  if not (logPath or console):
    logger.addHandler(logging.NullHandler())

  return logger


def float_array_string(arr):
  """
  format array of floats to 4 decimal places

  Args:
      arr: array of floats

  Returns:
      formatted string
  """
  return "[" + ", ".join(["{:.4f}".format(el) for el in arr]) + "]"
