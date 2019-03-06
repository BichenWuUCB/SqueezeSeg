#! /usr/bin/python2
# -*- coding: utf-8 -*-
"""
Clock function to take running time following Segmatch.
"""

import datetime

class Clock(object):
    def __init__(self):
        self.kSecondsToMiliseconds = 1000.0
        self.kMicrosecondsToMiliseconds = 0.001
        self.start()

    def start(self):
        self.real_time_start_ = datetime.datetime.now()

    def takeTime(self):
        seconds = (datetime.datetime.now() - self.real_time_start_).seconds
        useconds = (datetime.datetime.now() - self.real_time_start_).microseconds

        self.real_time_ms_ = (seconds*self.kSecondsToMiliseconds + useconds*self.kMicrosecondsToMiliseconds) + 0.5

    def getRealTime(self):
        return self.real_time_ms_

    def takeRealTime(self):
        self.takeTime()
        return self.getRealTime()
