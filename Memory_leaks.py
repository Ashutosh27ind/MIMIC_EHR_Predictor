# Author : Ashutosh Kumar for finding issue with memory :

import time

"""
from guppy3 import hpy

def dump_heap(h, i):

   
   @param h: The heap (from hp = hpy(), h = hp.heap())
   @param i: Identifier str
   
  
   print("Dumping stats at: {0}".format(i))
   print('Memory usage: {0} (MB)'.format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024))
   print("Most common types:")
   objgraph.show_most_common_types()
  
   print("heap is:")
   print("{0}".format(h))
  
   by_refs = h.byrcs
   print("by references: {0}".format(by_refs))
  
   print("More stats for top element..")
   print("By clodo (class or dict owner): {0}".format(by_refs[0].byclodo))
   print("By size: {0}".format(by_refs[0].bysize))
   print("By id: {0}".format(by_refs[0].byid))
"""

# Actual RAM used :
import resource
print("RAM in use now ")
print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

# Using mem_top :
from mem_top import mem_top
# From time to time:
print("Information from mem_top as below .....")
print(mem_top()) 

# Using object graph :
print("Using the object graph now.....")
import objgraph
print(objgraph.show_most_common_types(limit=20))

"""
# Uisng https://pyrasite.readthedocs.io/en/latest/MemoryViewer.html :
import os, meliae.scanner, platform

print("Using the pyrasite now... ")

if platform.system() == 'Windows':
    temp = os.getenv('TEMP', os.getenv('TMP', '/temp'))
    path = os.path.join(temp, 'pyrasite-%d-objects.json' % os.getpid())
else:
    path = '/tmp/pyrasite-%d-objects.json' % os.getpid()
meliae.scanner.dump_all_objects(path)

"""


