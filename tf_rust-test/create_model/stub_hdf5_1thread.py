import queue
import threading
import h5py
import numpy as np
import sys
import os
import time
from typing import Iterable, Iterator, List
import random
import copy

from tensorflow.keras.utils import to_categorical
import tensorflow as tf



LOCK_WORK = threading.Lock()
LOCK_RESULTS = threading.Lock()
LOCK_FILES = threading.Lock()
EXIT_FLAG = queue.Queue(1)





class Assembler:

    # Constructs a handler for multiple HDF files with the same structure
    # params:
    #   file_list: list of HDF5 filenames, we assume the structure is the following:
    #             {"traces": dataset,
    #             "metadata": {
    #                       "key": dataset,
    #                       "plaintext": dataset,
    #                       "masks": dataset
    #                       }}
    def  __init__(self, file_list: list(), trace_length, trace_offset, n_classes, fixed_shift_array, augment_shift_scale, ispredict, shuffle, byte_index):
        self.files = []
        self.sizes = []
        self.added_sizes = []
        self.len = 0
        self.trace_offset = trace_offset
        self.trace_length = trace_length
        self.n_classes = n_classes
        self.fixed_shift_array = fixed_shift_array
        self.augment_shift_scale = augment_shift_scale
        self.ispredict = ispredict
        self.shuffle = shuffle
        self.byte_index = byte_index

        for filename in file_list:
            try:
                self.files.append(h5py.File(filename))
            except:
                sys.exit(f"Unable to open file {filename}.")
        
        shape_traces = self.files[0]["traces"][0].shape
        for file in self.files:
            try:
                cur_size = file['traces'].shape[0]
                if shape_traces != file['traces'][0].shape:
                    sys.exit("The traces do not have the same shape")
                self.len += cur_size
                self.sizes.append(cur_size)
                self.added_sizes.append(self.len)
                
            except:
                sys.exit("Not all files have the same structure !")
        self.added_sizes[-1] -= 1
        self.len -= 1
        
        self.sizes = np.array(self.sizes)
        self.added_sizes = np.array(self.added_sizes)
        self._work_q = queue.Queue(-1)

    def __len__(self):
        return self.len
    
    def __getitem__(self, i):
        if isinstance(i, slice):
            if i.stop > self.len:
                raise IndexError
            if i.step != None:
                for index in range(i.start, i.stop, i.step):
                    self._work_q.put(index)
            else:
                for index in range(i.start, i.stop):
                    self._work_q.put(index)
        elif isinstance(i, Iterable):
            for index in i:
                if index >= self.len:
                    raise IndexError
                self._work_q.put(index)
        else:
            if i > self.len:
                raise IndexError
            self._work_q.put(i)
        
        l = len(self._work_q.queue)
        res = []
        while len(res) != l:
            res.append(self.get_single_item(self._work_q.get()))
            self._work_q.task_done()
        
        if self.ispredict == True:
            return np.stack(res)
        else:
            traces = [i[0] for i in res]
            labels = [i[1] for i in res]
            return np.stack(traces), np.stack(labels)
        
    def get_single_item(self, i: int):
        file_number = np.argmin(np.abs(self.added_sizes - i))
        if self.added_sizes[file_number] <= i:
            file_number += 1
        index = i - self.added_sizes[file_number] if file_number > 0 else i 

        if self.shuffle:
            augment_shift = np.random.randint(
                -self.augment_shift_scale, self.augment_shift_scale)
        else:
            augment_shift = 0
        rand_shift = self.fixed_shift_array[i]
        
        trace, label = self.files[file_number]['traces'][index] , self.files[file_number]['metadata']['key'][index]

        trace = trace[self.trace_offset+rand_shift +
                          augment_shift:self.trace_offset+rand_shift+self.trace_length+augment_shift]
        trace = np.reshape(trace, (-1, 1))
        trace = trace / 64
        label_value = label[self.byte_index]

        if self.ispredict == True:
            return np.reshape(trace, (-1,1))
        else:
            return np.reshape(trace, (-1,1)), to_categorical(label_value, num_classes=self.n_classes)

class DataloaderThread(threading.Thread):

    def __init__(self, assembly: Assembler, work_q: List, result_q: queue.Queue):
        threading.Thread.__init__(self)
        self.assembly = assembly
        self.work_q = work_q
        self.result_q = result_q

    def run(self):
        while len(self.work_q) > 0:

            #print(f"Thread: {self.name}, work len before: {len(self.work_q)}\n")
            index = self.work_q.pop(0)
            #print(f"Thread: {self.name}, work len after: {len(self.work_q)}\n")

            LOCK_FILES.acquire()
            data = self.assembly[index]
            LOCK_FILES.release()

            #print(f"Thread: {self.name}, result len before: {len(self.result_q.queue)}\n")
            self.result_q.put(data, timeout=5)
            #print(f"Thread: {self.name}, result len after: {len(self.result_q.queue)}\n")
        #print(f"{self.name}: work done.")

class Dataloader(tf.keras.utils.Sequence):

    def __init__(self, assembly: Assembler,indexes, batch_size: int, shuffle: bool, num_workers: int, buffer = 1):
        self.assembly = assembly
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.buffer = buffer
        self.indexes = indexes

        self.batches = [i for i in self.indexes]
        random.shuffle(self.batches)
        self.batches = [self.batches[i:min(i+self.batch_size, len(self.indexes))] for i in range(len(self.indexes)//self.batch_size)]
        self.length = len(self.batches)

        self.result_q = queue.Queue(self.num_workers+self.buffer)
        self.cur_i = 0
        
        self.start_work()

    def on_epoch_end(self):
        # 'Updates index_in after each epoch'
        self.batches = [i for i in self.indexes]
        random.shuffle(self.batches)
        self.batches = [self.batches[i:min(i+self.batch_size, len(self.indexes))] for i in range(len(self.indexes)//self.batch_size)]
        self.cur_i = 0

    def start_work(self):
        
        work_q = []
        for t in range(self.num_workers):
            if self.cur_i < self.length:
                to_do = self.batches[self.cur_i]
                work_q.append(to_do)
                self.cur_i += 1
            else:
                self.on_epoch_end()
                to_do = self.batches[self.cur_i]
                work_q.append(to_do)
                self.cur_i += 1

        self.thread = DataloaderThread(self.assembly, work_q, self.result_q)
        self.thread.start()

    def __len__(self):
        return len(self.batches)


    def __getitem__(self, index):
        # print(f"MainThread: Sleeping, waiting for results ...\n")
        if self.buffer-len(self.result_q.queue) > 0 and not self.thread.is_alive():
            self.start_work()
        while self.result_q.empty():
            time.sleep(0.001)
        
        #print(len(self.result_q.queue))

        data = self.result_q.get() 
        #print(f"MainThread: results {self.cur_i} arrived\n")
        return data
   


if __name__ == "__main__":
    root_path = os.path.dirname(__file__)
    dir = root_path+"/../ASCAD/ATMEGA_AES_v1/ATM_AES_v1_variable_key/ASCAD_data/ASCAD_databases/splitted/"

    files = []
    for d,_,fs in os.walk(dir):
        i = 0
        for f in fs:
            if i < 1000:
                files.append(d+"/"+f)
                i += 1
    
    #files = [root_path + "/../ASCAD/ATMEGA_AES_v1/ATM_AES_v1_fixed_key/ASCAD_data/ASCAD_databases/ATMega8515_raw_traces.h5"]


    test = Assembler(files)
    indexes = np.random.randint(0,60000,size=(2000,))
    times = []
    for i in range(10):
        t1 = time.time()
        traces = test[indexes]
        t2 = time.time()
        times.append(t2-t1)
    print(f"\nBase Assembler class (2000 items array query): {np.mean(times)}\n")

    dl = Dataloader(test, 2000, True, 8,3)
    times = []
    t1 = time.time()
    for i, data in enumerate(dl):
        t2 = time.time()
        times.append(t2-t1)
        if i > 99:
            break
        time.sleep(0.5) # do calculation
        t1 = time.time()
    print(f"Async Dataloader : {np.mean(times)}")