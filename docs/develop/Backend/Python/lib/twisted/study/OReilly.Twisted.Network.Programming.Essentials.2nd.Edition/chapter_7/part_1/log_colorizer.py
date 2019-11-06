# -*- coding:utf8 -*-
import sys


from twisted.python.log import FileLogObserver


class ColorizedLogObserver(FileLogObserver):
    def emit(self, eventDict):
        # Text color reconfig
        self.write('\033[0m')

        if eventDict['isError']:
            # Text color is red 조정하는 ANSI 확장 비트열(escape sequence)
            self.write('\033[91m')

        FileLogObserver.emit(self, eventDict)


def Logger():
    return ColorizedLogObserver(sys.stdout).emit
