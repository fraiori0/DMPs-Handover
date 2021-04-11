import numpy as np
import scipy as sp
import scipy.integrate as integrate
import quaternion as quat
import matplotlib.pyplot as plt
from math import pi, sin , cos
import joblib as jbl

ff = lambda t,w: np.sin(t*w)

class DMP_R2H:
    """
    Dynamical Movement Primitive

    A DMP is defined by the combination of two dynamical systems:\n
    1. Transformation system, which describes the evolution of the state\n
        (TAU^2)*xdd = ALPHA_X*(BETA*(G-x)-TAU*xd) + f\n
        with f a forcing term that vanish as the system moves toward its goal:
        f = fnl * s * (G-y0)\n
        and fnl a generic nonlinear term (usually learned)\n
    2. Canonical system, which describes the evolution of the timing variable:\n
        TAU*sd = -ALPHA_X*s\n
        This system can be modified, but s should monotonically converge from 1 to 0\n

    The state y is considered composed by [s, x, xd]
    """
    def __init__(self, TAU_0=1.0, ALPHA_X=1.0, ALPHA_S=1.0, ALPHA_G=1.0, ALPHA_D=1.0, ALPHA_E=1.0, dim=3):
        self.dim = dim
        self.set_parameters(TAU_0, ALPHA_X, ALPHA_S, ALPHA_G, ALPHA_D, ALPHA_E)
        self.s0 = 1.0
        self.x0 = np.zeros(dim)
        self.fnl = lambda t:0.0
        self.G = np.zeros(dim)
        # y = [s, x, xd, g, ex, d, dd]
        self.S_INDEX = [0]
        self.X_INDEX = range(1,dim+1)
        self.XD_INDEX = range(dim+1, 2*dim + 1)
        self.G_INDEX = range(2*dim+1, 3*dim + 1)
        self.EX_INDEX = range(3*dim+1, 4*dim + 1)
        self.D_INDEX = 4*dim + 1
        self.DD_INDEX = 4*dim + 2
    def set_parameters(self, TAU_0, ALPHA_X, ALPHA_S, ALPHA_G, ALPHA_D, ALPHA_E):
        self.TAU_0 = TAU_0
        self.ALPHA_X = ALPHA_X
        self.ALPHA_S = ALPHA_S
        self.ALPHA_G = ALPHA_G
        self.ALPHA_D = ALPHA_D
        self.ALPHA_E = ALPHA_E
    def set_gains(self,Kcd, Kce, Ktd, Kte):
        self.Kcd = Kcd
        self.Kce = Kce
        self.Ktd = Ktd
        self.Kte = Kte
    def set_goal(self, G):
        self.G = G.copy()
        self.force_fun_gen()
    def set_initial_state(self, s0, x0, xd0, D0, Dd0, EX0):
        if not (x0.shape[0]==xd0.shape[0]) :
            print('DMP error, set_initial_state(): x0 and xd0 dimensions don\'t match')
        elif not(x0.shape[0]==self.dim):
            print('DMP error, set_initial_state(): x0 and self.dim don\'t match')
        else:
            self.s0 = s0
            self.x0 = x0.copy()
            self.xd0 = xd0.copy()
            self.force_fun_gen()
            #
            self.g = self.G.copy()
            self.D0 = D0
            self.D = D0
            self.d0 = D0
            self.Dd0 = Dd0
            self.dd0 = Dd0
            self.EX = EX0.copy()
            self.ex = EX0.copy()
            # Return initial state
            self.y0 = np.concatenate((self.s0, self.x0, self.xd0, self.g, self.ex, self.d0, self.dd0), axis=None)
            return self.y0.copy()
    def set_fnl(self, fun):
        """Set the non-linear part of the forcing term

        Parameters
        ----------
        fun : function object,
            should accept t as an input, e.g. fnl(t)
        """
        self.fnl = fun
        self.force_fun_gen()
        
    def force_fun_gen(self):
        def force(t,s):
            return s*self.fnl(s)*(self.G-self.x0)
        self.f = force

    def sigmoid(self, x):
        return 1.0/(1.0 + np.exp(-self.sigm_steepness*(x+self.sigm_offset)))
    
    def set_sigmoid(self, steepness, offset):
        self.sigm_steepness = steepness
        self.sigm_offset = offset
    
    def exp_damping(self,x):
        return 1.0-np.exp(-15.0*x**3)
    
    def obstacle_avoidance_force_Hoffmann(self, obs, y, beta=6.4, gamma=1000):
        # Hoffmann et al. 2009
        ox = obs - y[self.X_INDEX]
        xd = y[self.XD_INDEX]
        theta = np.arccos((ox.T.dot(xd))/(np.linalg.norm(ox) * np.linalg.norm(xd) + 1.0e-8))
        r = np.cross(ox, xd)
        cs = cos(pi/4.0)
        sn = sin(pi/4.0)
        qr = np.quaternion(cs, sn*r[0], sn*r[1], sn*r[2])
        xd_perp = (qr * np.quaternion(0, xd[0], xd[1], xd[2]) * qr.conjugate()).vec
        #print(np.cross(xd_perp, xd))
        f_obs = gamma * theta * xd_perp * np.exp(-beta * theta)
        return f_obs
    
    def obstacle_avoidance_force_Skeletor(self, obs, y, threshold=0.5, K=80.0):
        ox = obs-y[dmp.X_INDEX]
        ox_d = np.linalg.norm(ox)
        f_obs = - K * ox * ( 1.0/(1.0 + np.exp(20.0*(ox_d-threshold))) )
        return f_obs

    def ode_fun(self, t, y):
        # low pass ilters for goal, position error and partner's distance from handover position
        gd = self.ALPHA_G*(self.G - y[self.G_INDEX])
        exd = self.ALPHA_E*(self.EX - y[self.EX_INDEX])
        dd = y[self.DD_INDEX]
        ddd = self.ALPHA_D*((self.ALPHA_D/4.0)*(self.D - y[self.D_INDEX]) - y[self.DD_INDEX])
        # self.TAU = self.TAU_0 + self.Kcd * y[self.D_INDEX]*np.exp(y[self.DD_INDEX])/self.D0 + self.Kce * y[self.EX_INDEX].dot(y[self.EX_INDEX])
        # self.TAUd = (self.Kcd * np.exp(y[self.DD_INDEX]) *(y[self.D_INDEX] *ddd + dd) / self.D0) + self.Kce * y[self.EX_INDEX] * exd
        self.TAU = self.TAU_0 * (1.0 + self.Kce * y[self.EX_INDEX].dot(y[self.EX_INDEX]) + self.Kcd*self.exp_damping(y[self.D_INDEX]/self.D0) * self.sigmoid(y[self.DD_INDEX]) ) 
        if self.TAU<0.0:
            print("ERROR: self.TAU less than 0 in ode_fun()")
        Ctd = -self.TAU * self.ALPHA_X * y[self.XD_INDEX] * self.Ktd * self.exp_damping(y[self.D_INDEX]/self.D0) * self.sigmoid(y[self.DD_INDEX])
        Ct = self.Kte * y[self.EX_INDEX] - Ctd * y[self.XD_INDEX] #- self.TAUd * self.TAU * y[self.XD_INDEX]#
        #
        sd = -y[self.S_INDEX]*self.ALPHA_S/(self.TAU)
        xd = y[self.XD_INDEX]
        xdd = (self.ALPHA_X*((self.ALPHA_X/4.0)*(self.G-y[self.X_INDEX]) - self.TAU*y[self.XD_INDEX])+ self.f(t,y[self.S_INDEX]) + Ct
            )/(self.TAU**2)
        # state y = [s, x, xd, g, ex, d, dd]
        yd =  np.concatenate((sd, xd,xdd,gd,exd,dd,ddd), axis=None)
        return yd
    
    def ode_fun_learning(self, t, y, xdd):
        # low pass filters for goal, position error and partner's distance from handover position
        gd = self.ALPHA_G*(self.G - y[self.G_INDEX])
        exd = self.ALPHA_E*(self.EX - y[self.EX_INDEX])
        dd = y[self.DD_INDEX]
        ddd = self.ALPHA_D*((self.ALPHA_D/4.0)*(self.D - y[self.D_INDEX]) - y[self.DD_INDEX])
        # self.TAU = self.TAU_0 + self.Kcd * y[self.D_INDEX]*np.exp(y[self.DD_INDEX])/self.D0 + self.Kce * y[self.EX_INDEX].dot(y[self.EX_INDEX])
        # self.TAUd = (self.Kcd * np.exp(y[self.DD_INDEX]) *(y[self.D_INDEX] *ddd + dd) / self.D0) + self.Kce * y[self.EX_INDEX] * exd
        self.TAU = self.TAU_0 * (1.0 + self.Kce * y[self.EX_INDEX].dot(y[self.EX_INDEX]) + self.Kcd*self.exp_damping(y[self.D_INDEX]/self.D0) * self.sigmoid(y[self.DD_INDEX]) )
        if self.TAU<0.0:
            print("ERROR: self.TAU less than 0 in ode_fun()")
        Ctd = -self.TAU * self.ALPHA_X * y[self.XD_INDEX] * self.Ktd * self.exp_damping(y[self.D_INDEX]/self.D0) * self.sigmoid(y[self.DD_INDEX])
        Ct = self.Kte * y[self.EX_INDEX] + Ctd #- self.TAUd * self.TAU * y[self.XD_INDEX]#
        #
        sd = -y[self.S_INDEX]*self.ALPHA_S/(self.TAU)
        xd = y[self.XD_INDEX]
        # xdd = (self.ALPHA_X*((self.ALPHA_X/4.0)*(self.G-y[self.X_INDEX]) - self.TAU*y[self.XD_INDEX]))/(self.TAU**2)
        ### state y = [s, x, xd, g, ex, d, dd]
        f_ideal = (self.TAU**2)*xdd - self.ALPHA_X*((self.ALPHA_X/4.0)*(self.G-y[self.X_INDEX]) - self.TAU*y[self.XD_INDEX])
        xi = (y[self.S_INDEX]*(self.G-self.x0))
        yd =  np.concatenate((sd, xd,xdd,gd,exd,dd,ddd), axis=None)
        return yd, f_ideal, xi

    def f_eq(self, t, y, f_ext=0.0):
        # low pass filters for goal, position error and partner's distance from handover position
        gd = self.ALPHA_G*(self.G - y[self.G_INDEX])
        exd = self.ALPHA_E*(self.EX - y[self.EX_INDEX])
        dd = y[self.DD_INDEX]
        ddd = self.ALPHA_D*((self.ALPHA_D/4.0)*(self.D - y[self.D_INDEX]) - y[self.DD_INDEX])
        # self.TAU = self.TAU_0 + self.Kcd * y[self.D_INDEX]*np.exp(y[self.DD_INDEX])/self.D0 + self.Kce * y[self.EX_INDEX].dot(y[self.EX_INDEX])
        # self.TAUd = (self.Kcd * np.exp(y[self.DD_INDEX]) *(y[self.D_INDEX] *ddd + dd) / self.D0) + self.Kce * y[self.EX_INDEX] * exd
        self.TAU = self.TAU_0 * (1.0 + self.Kce * y[self.EX_INDEX].dot(y[self.EX_INDEX]) + self.Kcd * self.exp_damping(y[self.D_INDEX]/self.D0) * self.sigmoid(y[self.DD_INDEX]) )
        if self.TAU<0.0:
            print("ERROR: self.TAU less than 0 in ode_fun()")
        Ctd = -self.TAU * self.ALPHA_X * y[self.XD_INDEX] * self.Ktd * self.exp_damping(y[self.D_INDEX]/self.D0) * self.sigmoid(y[self.DD_INDEX]) 
        Ct = self.Kte * y[self.EX_INDEX] + Ctd #- self.TAUd * self.TAU * y[self.XD_INDEX]#
        #
        sd = -y[self.S_INDEX]*self.ALPHA_S/(self.TAU)
        xd = y[self.XD_INDEX]
        xdd = (
            self.ALPHA_X*((self.ALPHA_X/4.0)*(self.G-y[self.X_INDEX]) - self.TAU*y[self.XD_INDEX])+ self.f(t,y[self.S_INDEX]) + Ct + f_ext
        )/(self.TAU**2)
        ### state y = [s, x, xd, g, ex, d, dd]
        f_eq = xdd*(self.TAU**2)
        yd =  np.concatenate((sd, xd,xdd,gd,exd,dd,ddd), axis=None)
        return yd, f_eq

    def integrate(self,tf, t_off = 0.0):
        t_int = np.array((0.0, tf)) + t_off
        self.ode_sol=integrate.solve_ivp(
            self.ode_fun, t_int, self.y0,
            dense_output = True
            )
    
    def integrate_Euler(self,y,t,dt):
        y_delta = self.ode_fun(t,y) * dt
        return y + y_delta
    
    def integrate_Euler_learning(self,y,xdd,t,dt):
        if y[self.D_INDEX]<0.0:
            y[self.D_INDEX]=0.0
        yd, f_ideal, xi = self.ode_fun_learning(t,y, xdd)
        y_delta = yd * dt
        return y + y_delta, f_ideal, xi
    
    def integrate_Euler_feq(self, y, t, dt, f_ext=0.0):
        yd, f_eq = self.f_eq(t,y, f_ext=f_ext)
        y_delta = yd*dt
        return y + y_delta, yd, f_eq

    def plot_sol(self, phases=True):
        fig_t,axs_t = plt.subplots(2,(self.x0.shape[0]+1))
        for index, elem in enumerate(self.x0):
            axs_t[0,index+1].plot(
                self.ode_sol['t'], self.ode_sol['y'][index+1],
                color="xkcd:teal"
                )
            axs_t[0,index+1].set(
                ylabel="x[%d] [m]"%index,
                xlabel="t [s]",
                title='Position'
                )
            axs_t[1,index+1].plot(
                self.ode_sol['t'], self.ode_sol['y'][self.x0.shape[0]+index+1],
                color="xkcd:dark teal"
                )
            axs_t[1,index+1].set(
                ylabel="xd[%d] [m/s]"%index,
                xlabel="t [s]",
                title='Velocity'
                )
        axs_t[0,0].plot(
            self.ode_sol['t'], self.ode_sol['y'][0],
            color="xkcd:salmon"
            )
        axs_t[0,0].set(
            ylabel="s",
            xlabel="t [s]",
            title='Position'
            )
        if phases:
            fig_ph, axs_ph = plt.subplots(2,self.x0.shape[0])
            for index, elem in enumerate(self.x0):
                axs_ph[0,index].plot(
                    self.ode_sol['y'][index+1], self.ode_sol['y'][self.x0.shape[0]+index+1],
                    color="xkcd:dark salmon"
                    )
                axs_ph[0,index].set(
                    xlabel="x[%d] [m]"%index,
                    ylabel="xd[%d] [m]"%index,
                    title='Phase plot x[%d]'%index
                    )
        plt.show()

### Minimum Jerk
def minimum_jerk(t, tf, x0, xf):
    # simple check on t and tf
    if ((t<0.0) or (tf<0.0)):
        print("Error in minimum_jerk: t or tf negative")
        raise ValueError
    t_rel = t/tf
    x   = x0 +               (xf-x0)*( 10.0*(t_rel**3) - 15.0*(t_rel**4) + 6.0*(t_rel**5) )
    xd  = (1.0/tf)*          (xf-x0)*( 30.0*(t_rel**2) - 60.0*(t_rel**3) + 30.0*(t_rel**4) )
    xdd = (1.0/tf)*(1.0/tf)* (xf-x0)*( 60.0*(t_rel)    - 180.0*(t_rel**2) + 120.0*(t_rel**3) )
    return x, xd, xdd

# Example of how to use this class

if __name__ == '__main__':
    TAU_0 = 1.0
    DMP_params_dict = {
            'TAU_0':    1.0,
            'ALPHA_X':  20.0,
            'ALPHA_S':  4.0,
            'ALPHA_G':  20.0,
            'ALPHA_D':  20.0,
            'ALPHA_E':  20.0
    }
    dmp = DMP_R2H(**DMP_params_dict)
    dmp.set_gains(
        Kcd=10.0,
        Kce=1000.0,
        Ktd=10.0,
        Kte=100.0,#1000.0,
    )
    dmp.set_sigmoid(
        steepness=1.0,
        offset=0.0
    )
    #
    G = np.array((0.7,0.0,0.0))
    dmp.set_goal(G)
    x0 = np.array((-0.3, -0.5, 0.0))
    #
    y0 = dmp.set_initial_state(
        s0 = 1.0,
        x0=x0,
        xd0=np.zeros(3),#np.random.rand(3)-0.5,
        D0=0.01,
        Dd0=0.0,
        EX0=np.zeros(3)
    )
    #
    fnl = lambda t: 0.0 # 0.2*np.sin(np.array((0.5,0.2,0.8))*t) #0.0
    model_name = './forcing_term_models/fnl_KNNR_minimum_jerk.joblib'
    loaded_model = jbl.load(model_name)
    def fnl(s):
        return loaded_model.predict(np.array(s).reshape(-1,1))
    dmp.set_fnl(fnl)
    dmp.force_fun_gen()
    #
    # Partner's hand distance from goal 
    d_fun = lambda t: (1.0 - np.sin(pi*(3*t-0.5)))
    d_fun_2 = lambda t: 0.0
    ex_fun = lambda t: 0.1*np.array((0.0,0.0,1.0))*(np.sin(pi*(3*t-0.5)))
    ex_fun_2 = lambda t: 0.0*np.array((0.0,0.0,1.0))*(np.sin(pi*(3*t-0.5)))
    # Simulate system
    dt = 1.0/300.0
    tf = 5.0
    t_array = np.arange(0.0,tf,dt)
    d_array = []
    ex_array = []
    sol = []
    E_array = []
    y0_start = y0.copy()
    for t in t_array:
        sol.append(y0)
        if t<1.0:
            dmp.D = d_fun(t)
            dmp.EX = ex_fun_2(t) #(y0[dmp.X_INDEX]-y0_start[dmp.X_INDEX])
        else: 
            dmp.D = d_fun_2(t)
            dmp.EX = ex_fun_2(t)
            #dmp.set_goal(np.array((0.0,0.0,1.306)))
        obs = np.array([0.0, 0.0, y0[dmp.X_INDEX][2]]) # obstacle always at the same height of the dmp position
        f_obs = dmp.obstacle_avoidance_force_Skeletor(obs, y0, K=0.0)
        y0, _, _ = dmp.integrate_Euler_feq(y0, t, dt, f_ext=f_obs)
        d_array.append(dmp.D)
        ex_array.append(dmp.EX.copy())
        E_array.append(0.5*dmp.TAU**2*y0[dmp.XD_INDEX].dot(y0[dmp.XD_INDEX]) + 0.5*((dmp.ALPHA_X**2)*0.25)*(y0[dmp.X_INDEX]-dmp.G).dot(y0[dmp.X_INDEX]-dmp.G))
    sol = np.array(sol)
    print(sol.shape)
    ex_array = np.array(ex_array)
    # dmp.integrate(tf=tf)
    # dmp.ode_fun(0.0, y0)
    # print(dmp.ode_fun(0.0, y0))
    # print(dmp.f(0.5,0.1))
    # dmp.plot_sol(phases=True)
    fig_t,axs_t = plt.subplots(2,4)
    for index in range(0,3):
        axs_t[0,index+1].plot(
            t_array, sol[:,dmp.X_INDEX[0]+index],
            color="xkcd:teal"
            )
        axs_t[0,index+1].set(
            #ylabel="x[%d] [m]"%index,
            xlabel="t [s]",
            title='x[%d]'%index
            )
        axs_t[1,index+1].plot(
            t_array, sol[:,dmp.XD_INDEX[0]+index],
            color="xkcd:dark teal"
            )
        axs_t[1,index+1].set(
            #ylabel="xd[%d] [m/s]"%index,
            xlabel="t [s]",
            title='xd[%d]'%index
            )
    # axs_t[0,1].plot(
    #     t_array, sol[:,dmp.X_INDEX[0]+2],
    #     color="xkcd:teal"
    #     )
    # axs_t[0,1].set(
    #     #ylabel="x[%d] [m]"%2,
    #     xlabel="t [s]",
    #     title='x[%d]'%2
    #     )
    # axs_t[1,1].plot(
    #     t_array, sol[:,dmp.XD_INDEX[0]+2],
    #     color="xkcd:dark teal"
    #     )
    # axs_t[1,1].set(
    #     #ylabel="xd[%d] [m/s]"%2,
    #     xlabel="t [s]",
    #     title='xd[%d]'%2
    #     )
    axs_t[0,0].plot(
        t_array, sol[:,dmp.S_INDEX],
        color="xkcd:salmon"
        )
    axs_t[0,0].set(
        ylabel="s",
        xlabel="t [s]",
        title='Phase'
        )
    axs_t[1,0].plot(
        t_array, ex_array,
        color="xkcd:red"
        )
    axs_t[1,0].plot(
        t_array, d_array,
        color="xkcd:red"
        )
    axs_t[1,0].set(
        ylabel="ex [m]",
        xlabel="t [s]",
        title='Distance d'
        )
    # axs_t[1,0].plot(
    #     t_array, ex_array,
    #     color="xkcd:light teal"
    #     )
    # axs_t[1,0].set(
    #     ylabel="e [m]",
    #     xlabel="t [s]",
    #     title='Error'
    #     )
    # axs_t[1,0].plot(
    #     t_array, E_array,
    #     color="xkcd:dark green"
    #     )
    # axs_t[1,0].set(
    #     ylabel="D [m]",
    #     xlabel="t [s]",
    #     title='Energy'
    #     )

    fig2, axs2 = plt.subplots(1,1)
    axs2.plot(
        sol[:,dmp.X_INDEX[0]], sol[:,dmp.X_INDEX[1]],
        color="xkcd:salmon"
        )
    plt.show()