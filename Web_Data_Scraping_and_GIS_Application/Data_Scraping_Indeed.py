# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 20:14:25 2017

@author: pradeep sathyamurthy

Course: GEO-441 - Geographic Information Systems

Data Scrapping from indeed website
"""

import urllib.request as url
from bs4 import BeautifulSoup as bs
import pandas as pd
import os
os.chdir('D:\Courses\GEO441 - GIS For Community Development\Project')

# Functions for scrapping data

def title_data(obj):
    bs_obj0 = bs(obj,'html.parser')
    job_tags = bs_obj0.find('a', {"data-tn-element": "jobTitle"})
    if job_tags:
        job_data1 = job_tags.text
        job_data = job_data1.replace(',','')
    else:
        job_data = 'NoData'
    return job_data.strip()

def com_data(obj):
    bs_obj1 = bs(obj,'html.parser')
    com_tags = bs_obj1.find('span', {"class": "company"})
    if com_tags:
        com_data = com_tags.text
        company_data = com_data.replace(',','')
    else:
        company_data = 'NoData'
    return company_data.strip()

def loc_data(obj):
    bs_obj2 = bs(obj,'html.parser')
    loc_tags = bs_obj2.find('span', {"class": "location"})
    if loc_tags:
        loc_data = loc_tags.text
    else:
        loc_data = 'NoData'
    return loc_data.strip()
    
def rev_data(obj):
    bs_obj2 = bs(obj,'html.parser')
    rev_tags = bs_obj2.find('span', {"class": "slNoUnderline"})
    #print(rev_tags)
    if rev_tags:
        rev_data = rev_tags.text        
        review_data = rev_data.replace(',','')
    else:
        review_data = 'NoData'
    return review_data

def whole_data(obj):
    whl_lst = list()
    bs_obj = bs(obj,'html.parser')
    whl_tags = bs_obj.findAll('div', {"data-tn-component": "organicJob"})
    #print(whl_tags)
    for tag in whl_tags:
        tag = str(tag)
        #print(tag)        
        title = title_data(tag)
        company = com_data(tag)
        review = rev_data(tag)
        location = loc_data(tag)
        single_div_tag_data = title + ',' + company + ',' + review + ',' + location
        whl_lst.append(single_div_tag_data)
    return whl_lst
    
# sending a get request to career builder site
# Site URL with job key is https://www.indeed.com/jobs?q=Data+Scientist&start=
# with pagination: https://www.indeed.com/jobs?q=Data+Scientist&start=10
# Pagination is a multiple of 10 in indeed, that is page 1 is referred as 10, which means the number of records retrived
# 10 records are retreived for a page
job_key = 'Data Scientist'
job_frame_key = job_key.strip().split()
job_url_key = 'jobs'+'?q='+job_frame_key[0]+'+'+job_frame_key[1]+'&'+'start='
#print(job_url_key)
final_url= 'https://www.indeed.com/' + job_url_key
print(final_url)
url_read = url.urlopen(final_url).read()
bs_obj = bs(url_read,'html.parser')
page_tag = bs_obj.find('div', {"id": "searchCount"})
page_tag_txt = page_tag.text
page_tag_data = page_tag_txt.split()
total_rec = int(page_tag_data[-1].replace(',',''))
total_pages = round(total_rec / 10)
print(total_rec)
print(total_pages)

# Data scrapping from each page from 1 to 10 = range(1,11)
final_data = list()
title_lst = list()
com_lst = list()
loc_lst = list()
review_lst = list()
whl_lst = list()
for i in range(0,total_pages,10):
    final_url_paginated = final_url + str(i)
    data_html_cb = url.urlopen(final_url_paginated).read()
    col0 = whole_data(data_html_cb)
    whl_lst.append(col0)
    final_data.append(data_html_cb)

# Building data for each row listed as part of the website
final_title_lst = list()
final_com_lst = list()
final_rev_lst = list()
final_city_lst = list()
final_state_lst = list()
final_country_lst = list()
city = ''
state = ''
county = ''
for lst in whl_lst:
    for item in lst:
        city = ''
        state = ''
        county = ''
        items = item.split(',')
        data0 = items[0].strip()
        data1 = items[1].strip()
        data2 = items[2].strip()
        if (len(items)==4):
            data3 = items[3].strip()
            if (data3.upper() == 'UNITED STATES'):
                country = 'United States'
            else:
                city = item[-1]
        elif (len(items)==5):
            country = 'United States'
            #state = items[-1].strip()
            state_data = items[-1].split()
            state = state_data[0]
            city = items[-2].strip()
        elif (len(items)>5):
            country = 'United States'
            city = items[-2]
            state_data = items[-1].split()
            state = state_data[0]
        final_title_lst.append(data0)
        final_com_lst.append(data1)
        final_rev_lst.append(data2)
        final_city_lst.append(city)
        final_state_lst.append(state)
        final_country_lst.append(country)

# Data count from each extract 
len(final_title_lst)   
len(final_com_lst)
len(final_rev_lst)
len(final_city_lst)
len(final_state_lst)
len(final_country_lst)

# Creating a data frame
dtf = pd.DataFrame({'Title':final_title_lst})
dtf['Company'] = final_com_lst
dtf['Review'] = final_rev_lst
dtf['City'] = final_city_lst
dtf['State'] = final_state_lst
dtf['Country'] = final_country_lst

# Create a csv file
dtf.to_csv('Indeed_Raw_Data_Scientist_09Aug2017.csv')

