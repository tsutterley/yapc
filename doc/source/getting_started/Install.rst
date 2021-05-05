======================
Setup and Installation
======================

Presently ``yapc`` is only available for use as a
`GitHub repository <https://github.com/tsutterley/yapc>`_.
The contents of the repository can be download as a
`zipped file <https://github.com/tsutterley/yapc/archive/main.zip>`_  or cloned.
To use this repository, please fork into your own account and then clone onto your system.

.. code-block:: bash

    git clone https://github.com/tsutterley/yapc.git

Can then install using ``setuptools``

.. code-block:: bash

    python setup.py install

or ``pip``

.. code-block:: bash

    python3 -m pip install --user .

Alternatively can install the utilities directly from GitHub with ``pip``:

.. code-block:: bash

    python3 -m pip install --user git+https://github.com/tsutterley/yapc.git
