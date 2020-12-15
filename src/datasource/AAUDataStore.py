import json

import smbclient

class AAUDataStore:
  def __init__(self):
    config = json.load(open('c.json', 'r'))
    self._connection = smbclient
    self._connection.ClientConfig(username=config['DATA_USER'], password=config['DATA_PASS'])

  @property
  def connection(self) -> smbclient:
      return self._connection

  @connection.setter
  def connection(self, value):
      pass
