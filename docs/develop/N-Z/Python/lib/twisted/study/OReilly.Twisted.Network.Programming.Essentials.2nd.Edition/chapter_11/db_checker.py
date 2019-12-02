# -*- coding:utf8 -*-
from zope.interface import implements

from twisted.cred import error
from twisted.cred.checkers import ICredentialsChecker
from twisted.cred.credentials import IUsernameHashedPassword
from twisted.internet.defer import Deferred


class DBCredentialsChecker(object):
    implements(ICredentialsChecker)
    credentialInterfaces = (IUsernameHashedPassword,)

    def __init__(self, runQuery, query):
        # dbpool = adbapi.ConnectionPool(DB-process-commands-name, DB-info)
        # self.runQuery = dbpool.runQuery
        self.runQuery = runQuery
        # self.query = 'DB Query'
        self.query = query

    def requestAvatarId(self, credentials):
        for interface in self.credentialInterfaces:
            if interface.providedBy(credentials):
                break
            else:
                raise error.UnhandledCredentials()

        dbDeferred = self.runQuery(self.query, (credentials.username,))
        d = Deferred()
        dbDeferred.addCallbacks(
            self._cbAuthenticate,
            self._ebAuthenticate,
            callbackArgs=(credentials, d),
            errbackArgs=(credentials, d)
        )
        return d

    def _cbAuthenticate(self, result, credentials, deferred):
        if not result:
            deferred.errback(error.UnauthorizedLogin('User not in Database.'))
        else:
            username, passwd = result[0]
            if credentials.checkPassword(passwd):
                deferred.callback(credentials.username)
            else:
                deferred.errback(error.UnauthorizedLogin('Passwd mismatch'))

    def _ebAuthenticate(self, failure, credentials, deferred):
        deferred.errback(error.LoginFailed(failure))
