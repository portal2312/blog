#!/usr/bin/env python
# -*- coding:utf8 -*-
import urllib2


def run():
    print urllib2.urlopen(url='http://localhost:8000').read()

if __name__ == '__main__':
    print 'non_blocking_client.py'
    run()
