# time-adaptive-control-4d-var
## Description
This project provides the code for the paper "A time adaptive optimal control approach for 4D-var data assimilation problems governed by parabolic PDEs" by Carmen Gräßle and Jannis Marquardt.

We interpret the 4D-var data assimilation problem for a parabolic partial differential equation (PDE) in the context of optimal control and revisit the process of deriving optimality conditions for an initial control problem. This is followed by a reformulation of the optimality conditions into an elliptic PDE, which is only dependent on the adjoint state and can therefore be solved directly without the need for e.g. gradient methods or related iterative procedures. Furthermore, we derive an a-posteriori error estimation for this system as well as its initial condition. We utilize this estimate to formulate a procedure for the creation of an adaptive grid in time for the adjoint state. This is used for 4D-var data assimilation in order to identify suitable time points to take measurements. 

The code - as it is given in this repository - can be used to recreate the numerical examples from the paper. Furthermore, it can easily be altered for the implementation of own tests.

> [!NOTE]
> The code has not been optimized for efficiency. The implementation concentrates on the extendability of the method for further research later on.

## System requirements
The code has been implemented and tested in `Python 3.9` on a MacBook M1 Pro with macOS 14.2. The following libraries/modules are required for the execution of the code:
- `sys`
- `matplotlib`
- `numpy`
- `scipy`
- `labellines`<br />
*(If the `labellines` library causes problems, you may also leave it away and use a simple legend in the `matplotlib` plot. Its only occurrences are in the `helper.py` file.)*

## Usage
The file and folder structure of this project is as follows:


## License
This project is licensed under the terms of the ATTRIBUTION-SHAREALIKE 4.0 INTERNATIONAL ([CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/deed.en)) license.

## Contact
For questions, suggestions and bugs, please contact<br/>
Jannis Marquardt

Institute for Partial Differential Equations<br/>
Technische Universität Braunschweig<br/>
Universitätsplatz 2<br/>
38106 Braunschweig<br/>
Germany

Mail: j.marquardt@tu-braunschweig.de <br/>
Web: https://www.tu-braunschweig.de/en/ipde/jmarquardt

