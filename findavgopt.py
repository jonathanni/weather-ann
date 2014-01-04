#!/usr/bin/python

f = open("outputs.txt", "r");
l = map(abs, map(float, f.read().split(" ")));
f.close()

print (sum(l) / len(l), max(l), min(l))
