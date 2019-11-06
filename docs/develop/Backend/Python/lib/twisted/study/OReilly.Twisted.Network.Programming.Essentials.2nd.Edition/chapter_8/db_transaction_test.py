# -*- coding:utf8 -*-
from twisted.internet import reactor
from twisted.enterprise import adbapi


dbpool = adbapi.ConnectionPool('sqlite3', 'hosts.db', check_same_thread=False)


def _createHostsTable(transaction, hosts):
    transaction.execute('''
    create table hosts(
        id int primary key,
        name varchar(127),
        ip varchar(63)
    )''')

    for id_, name, ip in hosts:
        print id_, name, ip
        transaction.execute('''
            insert into hosts (
                id, name, ip
            ) values (
                %d, \'%s\', \'%s\'
            )''' % (
                id_, name, ip
            )
        )


def createHostsTable(hosts):
    return dbpool.runInteraction(_createHostsTable, hosts)


def getName(name):
    return dbpool.runQuery('select * from hosts where name = \'%s\';' % name)


def printResult(results):
    for elt in results:
        print elt


def finish():
    dbpool.close()
    reactor.stop()


def run():
    hosts = [(
        1, 'server-master', '127.0.0.1'
    ), (
        2, 'server-slave', '10.1.1.1'
    )]
    d = createHostsTable(hosts)
    d.addCallback(lambda x: getName('server-master'))
    d.addCallback(printResult)

    reactor.callLater(1, finish)
    reactor.run()


if __name__ == '__main__':
    print 'ex.8-2 db_transaction_test.py'
    run()
