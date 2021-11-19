#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 16:03:52 2021

@author: isaaclera
"""

# small -> medium
# 5-25
# small -> large
# 5-25

# list_supported_requests = [5,20,25]
# c = 20

# 5-25 : 0.66
# 5-20 : 1 

# (5,24)= a.large

# fl
# 5-15: 1/5
# 5-18: 2/5  
# 5-19: 3/5
# 5-20: 4/5
# 5-25: 1     6flavour



# 5-20: 1/5
# 5-25: 1     3flvouar

# fl
# 5-15: 5/24 
# 5-18: 18/24
# 5-19: 19/24
# 5-20: 20/24
# 5-25: 25/24=1

list_supported_requests = [5,18,19,20,25]
names = ["small","m1","m2","m3","large"]
max_flavour_name = "large"


actual = 0
supported_CRequests=list_supported_requests[actual]
current_service = names[actual]


future = 2
supported_FRequests=list_supported_requests[future]
future_service = names[future]


current_requests = 20





print("%s -> %s"%(current_service,future_service))

# if current_requests <= supported_CRequests:
#     print("Why do an adapt?")
#     print("0")
#     print("Done")
if current_requests > max(list_supported_requests) and current_service == max_flavour_name:
    print("Impossible to improve better")
    print("0")
    print("DONE----")
    
            
if current_requests > max(list_supported_requests) and future_service==max_flavour_name:
    print("Best adapt flavour possible")
    print("1.0")
    print("DONE----")
    
predicate = lambda x: x >= current_requests
item = next(filter(predicate, list_supported_requests), None)
ix_future_flavour = list_supported_requests.index(supported_FRequests)
ix_current_flavour = list_supported_requests.index(supported_CRequests)

if item is not None: # There is a level that supports that request numbe
    ix_best_flavour = list_supported_requests.index(item)
    if ix_future_flavour == ix_best_flavour:
        print("Best movement")
        print("1.0")
        print("Done")
    if ix_future_flavour < ix_best_flavour:
        print("Worst movement")
        print("0.0")
        print("Done")
    if ix_future_flavour > ix_best_flavour:
        print("It's good but it can be better")
        print((len(list_supported_requests)-(ix_future_flavour-ix_best_flavour))/ len(list_supported_requests))
        print(ix_current_flavour-ix_future_flavour)
        print("Done")
if item is None:
    # we need to look for the best, we need the last
    ix_best_flavour = len(list_supported_requests)-1
    print("A")
    print(ix_future_flavour)
    print((ix_future_flavour+1) / len(list_supported_requests)) #distance to the best one
