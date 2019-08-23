import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import csv
import re
import pybrain.tools as tools

"""
Not zero based since the file format is not zero based
"""
class CrossSectionalDataProvider:
    def __init__(self,base_path,full_excel_path,norm=True):
        self.base_data_path = base_path 
        self.excel_path = full_excel_path
        self._full_data_path = '\OAS1_{0}_MR1\RAW\OAS1_{0}_MR1_mpr-1_anon.hdr'
        self._full_data_path_norm = '\OAS1_{0}_MR1\PROCESSED\MPRAGE\T88_111\OAS1_{0}_MR1_mpr_n{1}_anon_111_t88_masked_gfc.hdr'

        a = [x for x in csv.reader(open(self.excel_path))]

        self.norm = norm
        self.excel_nodes = a[0]
        self.excel_data = [x for x in a[1:]]

        None
    def _build_path_for_index(self,index,start=None):
        st = str(index)
        a = 4 - len(st)
        b = ''
        for x in range(0,a):
            b+='0'
        b+=st
        if self.norm:
            return self.base_data_path + self._full_data_path_norm.format(b,start)
        else:
            return self.base_data_path + self._full_data_path.format(b)

    def retrieve_full_data(self,index,start=3):        
        if self.norm:        
            try:
                epi_img1_data = nib.load(self._build_path_for_index(self.parse_name_for_number(index),start))
                data = epi_img1_data.get_data()[:,:,:,0]
                return data
            except:
                return self.retrieve_full_data(index,start + 1)
        else:
            epi_img1_data = nib.load(self._build_path_for_index(self.parse_name_for_number(index))) 
            return epi_img1_data.get_data()[:,:,:,0]


    def retrieve_full_data_example(self):        
        return self.retrieve_full_data(16)
    def get_data_with_CDR(self):
        list_index = []
        index = 0
        for x in self.excel_data:
            try:
                float(x[7])
                list_index.append((index))
                index+=1
            except:
                index+=1
                continue
        return list_index
    def parse_name_for_number(self,index):
        r = self.excel_data[index][0]
        st = ''.join(x for x in r if x.isdigit())
        return int(st[1:len(st) - 1])

    def get_CDR(self,index):
        try:
            a = float(self.excel_data[index][7])
            return a
        except:
            return None
    def get_gender(self,index):
        try:
            a = (self.excel_data[index][1])
            if a == "M":
                a = 1
            else:
                a = 0
            return a
        except:
            return None

def show_slices(slices):
    plt.imshow(slices[0],cmap='gray')
    plt.show()