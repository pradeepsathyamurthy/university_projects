# -*- coding: utf-8 -*-
"""
Created on Sun Aug  6 00:57:02 2017

@author: pradeep sathyamurthy

Course: GEO-441 - Geographic Information Systems

Data Scrapping from Career Builder website
"""

import urllib.request as url
from bs4 import BeautifulSoup as bs
import pandas as pd
import os
os.chdir('D:\Courses\GEO441 - GIS For Community Development\Project')

# Functions for scrapping data

def title_data(obj):
    job_lst = list()
    bs_obj0 = bs(obj,'html.parser')
    job_tags = bs_obj0.findAll('h2', {"class": "job-title hide-for-medium-up"})
    for tag in job_tags:
        job_data = tag.text
        print(job_data)
        job_lst.append(job_data)
    return job_lst

def pos_data(obj):
    pos_lst = list()
    bs_obj1 = bs(obj,'html.parser')
    pos_tags = bs_obj1.findAll('h4', {"class": "job-text employment-info"})
    for tag in pos_tags:
        pos_data = tag.text
        print(pos_data)
        pos_lst.append(pos_data)
    return pos_lst

def com_data(obj):
    com_lst = list()
    bs_obj2 = bs(obj,'html.parser')
    com_tags = bs_obj2.findAll('div', {"class": "columns large-2 medium-3 small-12"})
    for tag in com_tags:
        com_data = tag.text
        print(com_data)
        com_lst.append(com_data)
    return com_lst
    
def loc_data(obj):
    loc_lst = list()
    bs_obj = bs(obj,'html.parser')
    loc_tags = bs_obj.findAll('div', {"class": "columns end large-2 medium-3 small-12"})
    for tag in loc_tags:
        loc_data = tag.text
        print(loc_data)
        loc_lst.append(loc_data)
    return loc_lst


# sending a get request to career builder site
# Site URL with job key is http://www.careerbuilder.com/jobs-data-scientist?location=
#  with pagination: http://www.careerbuilder.com/jobs-data-scientist?page_number=1
job_key = 'data scientist'
job_frame_key = job_key.strip().split()
job_url_key = 'jobs'+'-'+job_frame_key[0]+'-'+job_frame_key[1]+'?'+'page_number='
#print(job_url_key)
final_url= 'http://www.careerbuilder.com/' + job_url_key
print(final_url)
url_read = url.urlopen(final_url).read()
bs_obj = bs(url_read,'html.parser')
page_tag = bs_obj.find('span', {"class": "page-count"})
page_tag_txt = page_tag.text
page_tag_data = page_tag_txt.split()
total_pages = int(page_tag_data[-1])
print(total_pages)

# Data scrapping from each page from 1 to 10 = range(1,11)
final_data = list()
title_lst = list()
pos_lst = list()
com_lst = list()
loc_lst = list()
for i in range(1,total_pages):
    final_url_paginated = final_url + str(i)
    #print(final_url_paginated)
    data_html_cb = url.urlopen(final_url_paginated).read()
    col0 = title_data(data_html_cb)
    col1 = pos_data(data_html_cb)
    col2 = com_data(data_html_cb)
    col3 = loc_data(data_html_cb)
    title_lst.append(col0)
    pos_lst.append(col1)
    com_lst.append(col2)
    loc_lst.append(col3)
    final_data.append(data_html_cb)

final_title_lst = list()
for obj in title_lst:
    for item in obj:
        data0 = item.strip()
        #print(data1)
        final_title_lst.append(data0)

final_pos_lst = list()
for obj in pos_lst:
    for item in obj:
        data1 = item.strip()
        #print(data1)
        final_pos_lst.append(data1)

final_com_lst = list()
for obj in com_lst:
    for item in obj:
        data2 = item.strip()
        #print(data2)
        final_com_lst.append(data2)
        
final_loc_lst = list()
for obj in loc_lst:
    for item in obj:
        data3 = item.strip()
        #print(data3)
        final_loc_lst.append(data3)

# Data count from each extract 
len(final_title_lst)   
len(final_pos_lst)
len(final_com_lst)
len(final_loc_lst)

state_lst = list()
address_lst = list()
addr = ''
for items in final_loc_lst:
    item = items.strip()
    item1 = item.split()
    if(len(item1)==2):
        print(item1)
        print(len(item1))
        state = item1[1]
        addr = item1[0]
        print(state)
        print(addr)
    elif(len(item1)==3):
        print(item1)
        print(len(item1))
        state = item1[-1]
        addr = item1[0] + item1[1]
        print(state)
        print(addr)
    elif(len(item1)==4):
        print(item1)
        print(len(item1))
        state = item1[-1]
        addr = item1[0] + item1[1] + item1[2]
        print(state)
        print(addr)
    elif(len(item1)==5):
        print(item1)
        print(len(item1))
        for item in item1:
            if len(item) == 2:
                state = item
            else:
                addr=addr+item
        #addr = ''
        print(state)
        print(addr)
    elif(len(item1)==6):
        print(item1)
        print(len(item1))
        for item in item1:
            if len(item) == 2:
                state = item
            else:
                addr=addr+item
        #addr = ''
        print(state)
        print(addr)
    state_lst.append(state)
    len(state_lst)
    address_lst.append(addr)
    len(address_lst)


# Creating a data frame
dtf = pd.DataFrame({'JOb_Title':final_title_lst})
dtf['Position'] = final_pos_lst
dtf['Company'] = final_com_lst
dtf['Location'] = final_loc_lst
dtf['State'] = state_lst
dtf['Address'] = address_lst

# # Create a csv file
dtf.to_csv('CB_Raw_Data_Scientist_8Aug.csv')


