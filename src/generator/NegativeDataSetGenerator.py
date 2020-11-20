import pandas as pd
import queue
import threading
import codecs


class WriteCsvThread(threading.Thread):
  def __init__(self, queue):
    threading.Thread.__init__(self)
    self.queue = queue

  def run(self):
    while True:
      result = self.queue.get()
      print(result, file=output)
      self.queue.task_done()


class ProcessCustomerIdThread(threading.Thread):
  def __init__(self, in_queue, out_queue):
    threading.Thread.__init__(self)
    self.in_queue = in_queue
    self.out_queue = out_queue

  def run(self):
    while True:
      id = self.in_queue.get()
      self.process(id)

      self.in_queue.task_done()

  def process(self, tid):
      existingProductIds = df[df.CUSTOMER_ID == tid].PRODUCT_ID.tolist()
      for productId in productIds:
        if productId not in existingProductIds:
          self.out_queue.put(str(tid) + ',' + str(productId))

# create shared queues
customerIdQueue = queue.Queue()
resultQueue = queue.Queue()

output = codecs.open('data/negative.csv', 'a')

# spawn threads to process each customer id
for i in range(0, 5):
  t = ProcessCustomerIdThread(customerIdQueue, resultQueue)
  t.setDaemon(True)
  t.start()

# spawn threads to write
t = WriteCsvThread(resultQueue)
t.setDaemon(True)
t.start()

#
df = pd.read_csv('data/sdata.csv')
df = df.drop(['MATERIAL', 'QUANTITY'], axis=1)
customerIds = df.CUSTOMER_ID.unique()
productIds = df.PRODUCT_ID.unique()

# add unique customer id to queue
for cid in customerIds:
  customerIdQueue.put(cid)

# wait for queue to get empty
customerIdQueue.join()
resultQueue.join()
