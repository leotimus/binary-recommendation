import threading
import time
import psutil
import csv
from matplotlib import pyplot
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, VPacker
import GPUtil

class benchThread (threading.Thread):
   def __init__(self, pollTime, active, filepath):
      threading.Thread.__init__(self)
      self.pollTime = pollTime
      self.active = active
      self.filepath = filepath
      self.daemon = True #useful to exit the logger when the main thread crashes  or is interrupted

   def run(self):
      timeCounter = 0
      print ("Starting " + self.name)
      with open(self.filepath, 'w+', newline='') as file:
         writer = csv.writer(file)
         writer.writerow(["Time (s)", "CPU %", "Memory MB", "GPU %"])
      while self.active:
         timeCounter += self.pollTime
         log_stats(self.pollTime, timeCounter, self.filepath)
      create_graph_from_csv(self.filepath)
      print ("Exiting " + self.name)

def log_stats(pollTime, timeCounter, filepath):
   time.sleep(pollTime)
   #print ("%s %.2f" % (psutil.cpu_percent(), psutil.virtual_memory().used/(1024**2)))
   with open(filepath, 'a', newline='') as file:
      writer = csv.writer(file)
      gpu = 0
      
      if len(GPUtil.getGPUs()) > 0:
         gpu = round(GPUtil.getGPUs()[0].load * 100, 1)
      
      writer.writerow([timeCounter, psutil.cpu_percent(), round(psutil.Process().memory_info().rss / (1024**2)), gpu])

def create_graph_from_csv (filepath):
   with open(filepath, 'r') as f:
      data = list(csv.reader(f))

   csvData = data[1:-1]
   memMB = [int(i[2]) for i in csvData]
   timeSeconds = [int(i[0]) for i in csvData]
   cpuPercent = [float(i[1]) for i in csvData]
   gpuPercent = [float(i[3]) for i in csvData]

   fig, ax1 = pyplot.subplots()

   color = 'tab:red'
   ax1.set_xlabel('time (s)')
   ax1.set_ylabel('RAM (MB)', color=color)
   ax1.plot(timeSeconds, memMB, color=color)
   ax1.tick_params(axis='y', labelcolor=color)

   ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

   color = 'tab:blue'
   color2 = 'tab:green'
   # ax2.set_ylabel('CPU %  GPU %', color=color)  # we already handled the x-label with ax1
   ax2.plot(timeSeconds, cpuPercent, color=color)
   ax2.plot(timeSeconds, gpuPercent, color=color2)
   # ax2.tick_params(axis='y', labelcolor='tab:orange')

   ybox1 = TextArea("CPU % ", textprops=dict(color=color, size=10, rotation=90, ha='left', va='bottom'))
   ybox3 = TextArea("GPU % ", textprops=dict(color=color2, size=10, rotation=90, ha='left', va='bottom'))
   ybox = VPacker(children=[ybox1, ybox3], align="bottom", pad=0, sep=5)
   anchored_ybox = AnchoredOffsetbox(loc=8, child=ybox, pad=0., frameon=False, bbox_to_anchor=(1.1, 0.4),
                                     bbox_transform=ax2.transAxes, borderpad=0.)
   ax2.add_artist(anchored_ybox)

   # fig.tight_layout()  # otherwise the right y-label is slightly clipped

   # pyplot.show()
   gayStuff = 'top.csv'
   graphPath = filepath + '.graph.png'
   pyplot.savefig(graphPath)

# Create new threads

#thread1 = benchThread(1, "Thread-1", 1, 1)
#thread2 = benchThread(2, "Thread-2", 2, 1)
#thread3 = threading.Thread(target=print_time("pesho", 3, 4, 1))

# Start new Threads
#thread1.start()
#thread2.start()
#thread1.join()
#thread2.join()
#thread3.start()
#thread3.join()
#time.sleep(2)
#thread1.active = 0
#print ("Exiting Main Thread")