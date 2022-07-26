#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Fri Jun  5 10:41:56 2020

@author: zhxm
"""

from debugger import prev_state

class Foo:
    def __init__(self, x):
        self.x = x
    
    @prev_state(10, KeyError)
    def test1(self, idx):
        return self.x[idx]
    
    @prev_state(2, KeyError)
    def test2(self, idx):
        return self.x[idx]
    
    @prev_state(2, TypeError)
    def test3(self, idx):
        return self.x[idx]


foo = Foo({'a':1, 'c':3})
print(foo.test1('a1'))
print(foo.test1('b'))
print(foo.test1('a'))
print(foo.test1('b'))
print(foo.test1('c'))

print(foo.test2('c'))
print(foo.test2('b'))
print(foo.test2('a'))

foo.test3('d')
