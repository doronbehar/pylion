Simulation
==============

When using pylion, molecular dynamics simulations are offloaded from python to
LAMMPS.
Simulations are configured by appending parameters on a ``Simulation`` object that once setup and executed, generates an input file that is used by a LAMMPS subprocess.
An asynchronous update loop monitors the progress of the subprocess, allowing error interception and diagnostic updates.
Once complete (or on user interrupt) pylion terminates the LAMMPS subprocess is terminated and reverts control to you so you can analyse the generated data.

.. module:: pylion

Simulation object
-----------------

At the heart of pylion is the ``Simulation`` class.
It is implemented as a subclass of ``list`` so it provides a familiar API.


.. autoclass:: Simulation
  :members:
  :special-members: __contains__

  ``Simulation`` is for all practical purposes a list that only contains python dictionaries; appending any other type will raise an error.
  During its lifetime it will do the following:

  - check the various configuration parameters, fixes, and commands as you append them.
  - generate a ``simulation.lammps`` file using a jinja2 template.
  - call the ``lammps`` subprocess with said file and deal with output piping and signal handling.
  - generate an h5 file with all the necessary parameters needed to rerun the simulation.

  The same simulation will not run again if it has executed already.
  This keeps the simulations atomic and you can be sure that one h5 file correponds to one simulation.

  ``Simulation`` also keeps track of the unique ids of the appended dictionaries.
  All simulation items are identified through a ``uid`` if they have one, otherwise it defaults to ``None``.
  To search, index, or remove an item it needs to have a ``uid``.

  .. warning::
    Not all list methods are overridden or needed. Only the ones referenced here.



Attributes
----------

The simulation attributes are implemented as a simple subclass of ``dict`` that adds two methods to ``save`` and ``load`` the dictionary items to an h5 file. ``save`` first serialises the dict values as json strings so that they can be saved as h5 attributes. ``loads`` deserialises the json and returns proper python objects so you never know what happened.

A ``Simulation()`` defines a set of default attributes that control simulation parameters:

- *executable*, path to lammps binary. Can also be set to a string with spaces, to include multiple arguments. It can even be: ``mpirun lmp`` to parallelise the execution with `MPI <https://en.wikipedia.org/wiki/Message_Passing_Interface>`, as `explained in lammps docs <https://docs.lammps.org/Speed_omp.html#run-with-the-openmp-package-from-the-command-line>`.
- *gpu*, if not ``None``, set the arguments to the ``package gpu`` command as a single string. Check out lammps documentation for the `package <https://docs.lammps.org/package.html>`_ command for more details.
- *timestep*, the equations of motion are propagated by this much at every step.
  You can set this parameter to whatever value you want but ideally it would be faster than the fastest timescale in your problem (usually the rf frequency of the Paul trap).
  Any fix can also set the timestep automatically if it has a ``timestep`` key in its dictionary, whose value is less than the current value of the simulation timestep.
- *domain*, defines the lower limits of the spatial region of the simulation.
  The simulation box may expand beyond these initial limits, but all ions must be initially placed within this region or an error will be thrown by lammps.
  Check out the lammps documentation for the `boundary <http://lammps.sandia.gov/doc/boundary.html>`_ and `box <http://lammps.sandia.gov/doc/create_box.html>`_ commands for more information.
- *thermo_styles*, simply arguments to the `thermo_style <https://docs.lammps.org/thermo_style.html>`_ LAMMPS command. Useful if you want to see the temperature of the whole system while the simulation is running, and in the log file. Defaults to: ``['step', 'cpu']``.

  A guideline for optimisation is to ensure the simulation domain fits the
  enclosed atoms tightly; this ensures that when the domain is spatially
  partitioned there are an approximately equal number of ions per processor.
  When reducing the simulation box, lammps raises the following error::

      Cannot use neighbor bins - box size << cutoff

- *name*, slugified name of the simulation. Also used for the h5 file.
- *neighbour*, skin and list.
  Check out the `neighbour <http://lammps.sandia.gov/doc/neighbor.html>`_ command and the ``nsq`` style to build neighbour lists.
  This is the default style used in pylion although its scaling is proportional to number of ions per processor squared.
- *coulombcutoff*, the range of the Coulomb interaction defaults to 10cm.
  You can reduce the neighbour skin size or the Coulomb cutoff to increase the simulation speed, but this may result in unphysical ion-ion interactions.
- *template*, the jinja2 template used.
- *version*, the pylion version.
- *rigid*, groups that are tagged as rigid.
  This is autogenerated by ion dictionaries that have the key ``rigid``.
  Defaults to a single element ``exists = False``.

At execution time the following attributes are also set:

- *time*, the time the simulation was started.
- *output_files*, names of the files used by ``dump`` commands to save the simulation output.

.. warning::
  Make sure you know what you are doing if you overwrite the defaults.

You can add literally anything to the above using dict notation.
Use it to add notes about the simulation, keep track of other parameters or anything you think belongs to the generated h5 file.
