# -*- coding:utf8 -*-
from twisted.internet import reactor
from twisted.enterprise import adbapi

# ERROR: 아래와 같을 때 추가 adbapi.ConnectionPool(, check_same_thread=False)
#   sqlite3.ProgrammingError: SQLite objects created in
#   a thread can only be used in that same thread.
#   The object was created in thread id 140104115697408 and
#   this is thread id 140104342914880
#
#   https://twistedmatrix.com/trac/ticket/3629
dbpool = adbapi.ConnectionPool('sqlite3', 'users.db', check_same_thread=False)


def getName(where=''):
    return dbpool.runQuery('select * from users %s;' % where)


def printResults(results):
    for elt in results:
        print elt


def finish():
    dbpool.close()
    reactor.stop()


def run():
    d = getName('where email = \'portal2312@gmail.com\'')
    d.addCallback(printResults)
    reactor.callLater(1, finish)
    reactor.run()


if __name__ == '__main__':
    print 'ex.8-1 db_test.py'
    run()
