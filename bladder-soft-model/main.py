#Libraries
import autograd.numpy as np


#Local imports
from parameters import *
from functions import *
from plotting import *

#Initial conditions
VB0 = 5e-2 #[ml]
TB0 = 1e-2 #[Pa]
NBa0 = 1e-2 #[muV]

#Initial conditions
ODE0=[VB0,TB0,NBa0]

#Time range
samples_per_minute = 2000
t_final = 15 #[min]
t=np.linspace(0,t_final,t_final*samples_per_minute)



#integrate ODE
sol = odeint(ODE, ODE0, t)

VBs = sol[:, 0]
TBs = sol[:, 1]
NBas = sol[:, 2]

plot_vs_time('TB',t, sol)
