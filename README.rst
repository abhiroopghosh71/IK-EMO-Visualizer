IK-EMO Visualizer
==============================================================

The open source IK-EMO-Viz framework allows users to directly plug in data obtained from a multi-objective optimization 
problem. The framework leverages the powerful Dash framework to present innovization analytics from the supplied data.

.. _Installation:

Installation
********************************************************************************

It is recommended to create a separate Python environment for IK-EMO-Viz. Anaconda3 or Miniconda3 is recommended.

To get the latest source code:

.. code-block:: bash

    git clone https://github.com/abhiroopghosh71/IK-EMO-Visualizer.git
    cd IK-EMO-Visualizer
    pip install -r requirements.txt

Then, generate the sample data files:

.. code-block:: bash

    python utils/sample_data.py

.. _Usage:

Usage
********************************************************************************

Welded beam optimization problem data:

.. code-block:: bash

    python main.py --X-file "data/welded_beam/X.DAT" --F-file "data/welded_beam/F.DAT" --params-file "data/welded_beam/params.DAT" --port 8050

2D-truss optimization problem data:

.. code-block:: bash

    python main.py --X-file "data/truss2d/X.DAT" --F-file "data/truss2d/F.DAT" --params-file "data/truss2d/params.DAT" --port 8051

User-defined optimization problem data:

IK-EMO-Viz requires three files:

#. X.DAT, which consists of the decision variables for multiple solutions arranged in a matrix form.
#. F.DAT, which consists of the objective values for the solutions defined in X.DAT.
#. params.DAT, a JSON file with different problem parameters like number of variables, number of objectives, variable limits, etc. Refer to the generated params.DAT files in data/welded_beam and data/truss2d folders.

Once the necessary files are ready, run the code given below.

.. code-block:: bash

    python main.py --X-file "<path-to-X.DAT>" --F-file "<path-to-F.DAT>" --params-file "<path-to-params.DAT>" --port <desired-port>

