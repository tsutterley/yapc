============
utilities.py
============

Management utilities for Yet Another Photon Classifier (YAPC)

`Source code`__

.. __: https://github.com/tsutterley/yapc/blob/main/yapc/utilities.py


General Methods
===============

.. method:: yapc.utilities.build_logger(name, **kwargs)

    Builds a logging instance with the specified name

    Arguments:

      ``name``: name of the logger

    Keyword arguments:

      ``format``: event description message format

      ``level``: lowest-severity log message logger will handle

      ``propagate``: events logged will be passed to higher level handlers

      ``stream``: specified stream to initialize StreamHandler
