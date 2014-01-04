#!/usr/bin/python

for i in open("error.log", "r"):
  if i.startswith('[I] Total error:'):
    print i.split(' ')[-1][:-1];
