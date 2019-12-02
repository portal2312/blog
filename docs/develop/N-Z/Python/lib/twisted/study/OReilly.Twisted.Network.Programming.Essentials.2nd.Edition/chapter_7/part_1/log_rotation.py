# -*- coding:utf8 -*-
from twisted.python import log
from twisted.python import logfile

import os


def run():
    file_name = 'log_rotation.log'
    file_dir = os.path.abspath(os.path.curdir)

    # 매 100 bytes 회전하며 기록
    # file_dir/file_name.1, ... file_dir/file_name.n
    f = logfile.LogFile(file_name, file_dir, rotateLength=100)
    log.startLogging(file=f)
    log.msg('First msg')

    # 수동으로 회전한다.
    f.rotate()

    for i in range(5):
        log.msg('TEST msg', i)

    log.msg('Last msg')


if __name__ == '__main__':
    print 'ex.7-3 log_rotation.py'
    run()
