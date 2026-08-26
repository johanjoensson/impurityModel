Welcome to impurityModel's documentation!
=========================================

:Web Site:     https://github.com/johanjoensson/impurityModel
:Date:         |today|

``impurityModel`` is an exact-diagonalization solver for Anderson impurity models,
implemented in Python with performance-critical kernels in C++/Cython and parallelized
over MPI. It computes ground states, x-ray spectra (PS, XPS, XAS, NIXS, RIXS), and impurity
self-energies for DMFT-style workflows.

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   intro
   user_guide
   input_format

.. toctree::
   :maxdepth: 2
   :caption: How it works

   architecture_overview
   gf_solver_architecture
   basis_and_restrictions
   ground_state_report
   mpi_model
   greens_function_theory

.. toctree::
   :maxdepth: 1
   :caption: Reference

   configuration
   known_issues
   sphinx/_doc_build/impurityModel


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
