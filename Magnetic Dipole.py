import numpy as np
import matplotlib.pyplot as plt

'''
    We are going to compare two different particle pushers
    to see the motion of a charge in uniform magnetic field
    We use a particle class that stores a particles's charge,
    mass, position and velocity
'''
class Particle:
    def __init__(self, mass=1, charge=1, pos=np.array([0.,0.,0.]), 
                vel=np.array([0.1, 0.1, 0.])):
        self.m = mass
        self.q = charge
        self.pos = pos
        self.vel = vel
        return

'''
    Electric field as a function of position
'''
def E(x: np.array):
    return np.array([0., 0., 0.])

'''
    Magnetic field as a function of position
    Constant z field
'''
def B(x: np.array):  
    r = np.sqrt(x[0]**2 + x[1]**2 + x[2]**2)  
    theta = np.arctan(np.sqrt(x[0]**2 + x[1]**2) / x[2])  
    fai = np.arctan2(x[1], x[0])
    M = 400  
    Bx = M * (2 * np.cos(theta) * np.sin(theta) * np.cos(fai) + np.sin(theta) * np.cos(theta) * np.cos(fai)) / r**3  
    By = M * (2 * np.cos(theta) * np.sin(theta) * np.sin(fai) + np.sin(theta) * np.cos(theta) * np.sin(fai)) / r**3  
    Bz = M * (2 * np.cos(theta)**2 - np.sin(theta)**2 ) / r**3  
    return np.array([Bx, By, Bz])


'''
    Vanilla Partile Pusher that uses a leapfrom scheme
    and the initial velocity to get the momentum update
'''
def PushParticleSimple(particle: Particle, dt: float):
    x_old = particle.pos
    v_old = particle.vel
    a_old = (particle.q / particle.m) * (E(x_old) + np.cross(v_old, B(x_old)))

    x_new = x_old + v_old * dt + 0.5 * a_old * dt * dt
    a_new =  (particle.q / particle.m) * (E(x_new) + np.cross(v_old, B(x_new)))
    v_new = v_old + 0.5 * (a_old + a_new) * dt

    particle.pos = x_new
    particle.vel = v_new
    return


'''
    Non Relativistic Boris particle Pusher
'''
def PushParticleBoris(particle: Particle, dt: float):
    x_old = particle.pos
    v_old = particle.vel

    x_mid = x_old + 0.5 * v_old * dt

    v_minus = v_old + 0.5 * (particle.q / particle.m) * E(x_mid) * dt
    t_vec = 0.5 * (particle.q / particle.m) * B(x_mid) * dt
    t_mag = np.linalg.norm(t_vec)
    s_vec = 2 * t_vec / (1 + t_mag * t_mag)
    v_plus = v_minus + np.cross(v_minus + np.cross(v_minus, t_vec), s_vec)
    v_new = v_plus + 0.5 * (particle.q / particle.m) * E(x_mid) * dt

    x_new = x_mid +  0.5 * v_new * dt

    particle.pos = x_new
    particle.vel = v_new
    return

'''
    Relativistic Boris Pusher
'''
def PushParticleRelativisticBoris(particle: Particle, dt: float):
    x_old = particle.pos
    v_old = particle.vel

    x_mid = x_old + 0.5 * v_old * dt

    v_minus = v_old + 0.5 * (particle.q / particle.m) * E(x_mid) * dt
    v_minus_norm = np.linalg.norm(v_minus)
    gamma_minus = 1 / np.sqrt(1 - v_minus_norm**2)

    t_vec = 0.5 * (particle.q / particle.m) * B(x_mid) * dt / gamma_minus
    t_mag = np.linalg.norm(t_vec)
    s_vec = 2 * t_vec / (1 + t_mag * t_mag)
    v_plus = v_minus + np.cross(v_minus + np.cross(v_minus, t_vec), s_vec)
    v_new = v_plus + 0.5 * (particle.q / particle.m) * E(x_mid) * dt

    x_new = x_mid +  0.5 * v_new * dt

    particle.pos = x_new
    particle.vel = v_new
    return

'''
    Relativisitc Vay Pusher
'''
def PushParticleVay(particle: Particle, dt: float):
    x_old = particle.pos
    v_old = particle.vel
    v_old_norm = np.linalg.norm(v_old)
    gamma_old = 1 / np.sqrt(1 - v_old_norm**2)
    u_old = v_old * gamma_old

    x_mid = x_old + 0.5 * v_old * dt
    u_mid = (gamma_old * v_old) + particle.q * dt / (2 * particle.m) * (E(x_mid) + np.cross(v_old, B(x_mid)))    

    u_prime = u_mid + (particle.q * dt / (2 * particle.m)) * E(x_mid)
    tau = (particle.q * dt / (2 * particle.m)) * B(x_mid)
    u_star = np.dot(u_prime, tau)
    gamma_prime = np.sqrt(1 + np.linalg.norm(u_prime)**2)

    sigma = gamma_prime**2 - np.dot(tau, tau)
    gamma_new = np.sqrt( (sigma + np.sqrt(sigma**2 + 4 * (np.dot(tau, tau) + u_star*u_star))) /2)
    t_vec = tau / gamma_new

    s = 1 / np.sqrt(1 + np.dot(t_vec, t_vec))
    u_new = s * (u_prime + np.dot(u_prime, t_vec)*t_vec + np.cross(u_prime, t_vec))

    #v_bar = (u_old / gamma_old + u_new / gamma_new) / 2

    v_new = u_new / gamma_new
    x_new = x_mid +  0.5 * v_new * dt

    particle.pos = x_new
    particle.vel = v_new
    return

'''
    Higuera-Cary Pusher
'''
def PushParticleHC(particle: Particle, dt: float):
    x_old = particle.pos
    v_old = particle.vel
    v_old_norm = np.linalg.norm(v_old)
    gamma_old = 1 / np.sqrt(1 - v_old_norm**2)
    u_old = v_old * gamma_old

    x_mid = x_old + 0.5 * v_old * dt

    u_minus = u_old + (particle.q * dt / (2 * particle.m)) * E(x_mid)
    gamma_minus = np.sqrt(1 + np.dot(u_minus, u_minus))
    tau = (particle.q * dt / (2 * particle.m)) * B(x_mid)

    u_star = np.dot(u_minus, tau)
    sigma = gamma_minus**2 - np.dot(tau, tau)
    gamma_plus = np.sqrt((sigma + np.sqrt(sigma**2 + 4 * (np.dot(tau, tau) + u_star*u_star))) / 2)

    t_vec = tau / gamma_plus
    s = 1 / (1 + np.dot(t_vec, t_vec))
    u_plus = s * (u_minus + np.dot(u_minus, t_vec) * t_vec + np.cross(u_minus, t_vec))

    u_new = u_plus + (particle.q * dt / (2 * particle.m) * E(x_mid) + np.cross(u_minus, t_vec))
    gamma_new = np.sqrt(1 + np.dot(u_new, u_new))

    v_new = u_new / gamma_new
    x_new = x_mid +  0.5 * v_new * dt

    particle.pos = x_new
    particle.vel = v_new  
    return

'''
    Great us update two identical particles - one with Boris and 
    the other one with the vanilla pusher
'''
x_init = np.array([0., 0., 0.])
v_init = np.array([0.1, 0.1, 0.])

p1 = Particle(pos=x_init, vel=v_init)
p2 = Particle(pos=x_init, vel=v_init)
p3 = Particle(pos=x_init, vel=v_init)
p4 = Particle(pos=x_init, vel=v_init)
p5 = Particle(pos=x_init, vel=v_init)
dt = 0.1

'''
    The main function to run
'''
def main():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x1, y1, z1 = [], [], []
    x2, y2, z2 = [], [], []
    x3, y3, z3 = [], [], []
    x4, y4, z4 = [], [], []
    x5, y5, z5 = [], [], []
    for i in range(100000):
        PushParticleSimple(p1, dt)
        PushParticleBoris(p2, dt)
        PushParticleRelativisticBoris(p3, dt)
        PushParticleVay(p4, dt)
        PushParticleHC(p5, dt)
        x1.append(p1.pos[0]); y1.append(p1.pos[1]); z1.append(p1.pos[2])
        x2.append(p2.pos[0]); y2.append(p2.pos[1]); z2.append(p2.pos[2])
        x3.append(p3.pos[0]); y3.append(p3.pos[1]); z3.append(p3.pos[2])
        x4.append(p4.pos[0]); y4.append(p4.pos[1]); z4.append(p4.pos[2])
        x5.append(p5.pos[0]); y5.append(p5.pos[1]); z5.append(p5.pos[2])

    # Plot particle trajectories
    #L1 = ax.plot(x1, y1, z1, color='k', label="simple pusher")
    L2 = ax.plot(x2, y2, z2, color='b', label="boris pusher")
    L3 = ax.plot(x3, y3, z3, color='r', label="relativistic boris pusher")
    #L4 = ax.plot(x4, y4, z4, color='g', label="relativistic vay pusher")
    #L5 = ax.plot(x5, y5, z5, color='y', label="relativistic HC pusher")
    plt.legend()
    plt.show()

    return


if __name__ == "__main__": main()
