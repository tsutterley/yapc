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


.. method:: yapc.utilities.convert_arg_line_to_args(arg_line)

    Convert file lines to arguments

    Arguments:

        ``arg_line``: line string containing a single argument and/or comments


.. method:: yapc.utilities.copy(source, destination, move=False)

    Copy or move a file with all system information

    Arguments:

        ``source``: source file

        ``destination``: copied destination file

    Keyword arguments:

        ``move``: remove the source file
