import json
import os

import smbclient

class DataStore:
  def __init__(self):
    config = json.load(open('c.json', 'r'))
    self._aauDataStore = smbclient
    self._aauDataStore.ClientConfig(username=config['DATA_USER'], password=config['DATA_PASS'])

  def openFile(self, path:str, mode:str = 'r'):
    if os.path.isfile(path):
      return open(path, mode=mode)
    else:
      return self.aauDataStore.open_file(path, mode)

  @property
  def aauDataStore(self) -> smbclient:
      return self._aauDataStore

  @aauDataStore.setter
  def aauDataStore(self, value):
      pass
