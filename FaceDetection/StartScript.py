# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 17:16:46 2020

@author: user
"""

from ImageRetrievalUI import main

import time
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

limit_str = 'Wed Apr 29 23:59:59 2020'
limit = time.mktime(time.strptime(limit_str))

if __name__ == '__main__':
    if time.time() > limit:
        print ('Demo time expired!')
    else:
        main()
    
    

            
    
#    exec(open("./ImageRetrievalUI.py").read())
