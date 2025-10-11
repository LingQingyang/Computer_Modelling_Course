import numpy as np
import matplotlib.pyplot as plt

class Cosmology:
    def __init__(self, H0, Omega_m, Omega_lambda):
        self.H0 = H0
        self.Omega_m = Omega_m
        self.Omega_lambda = Omega_lambda
        self.Omega_k = 1.0 - self.Omega_m - self.Omega_lambda

    #"__init__" is a magic method in Python,  
    # which is called to initialize the parameters of a newly created object.

    #"self" refers to the object itself.

    #Compute the integrand of the distance formula
    def distance_integrand(self, z):
        """Calculate the integrand for the comoving distance."""
        return 1.0 / np.sqrt(
            self.Omega_m * (1 + z)**3 + 
            self.Omega_k * (1 + z)**2 + 
            self.Omega_lambda
            )
 
    def whether_flat(self):
        """Check if the universe is flat."""
        return self.Omega_k == 0.0
    
    def modify_Omega_lambda(self):
        """Modify Omega_lambda while keeping the curvature of the Universe the same."""
        self.Omega_lambda = 1.0 - self.Omega_k - self.Omega_m

    def modify_Omega_m(self):
        """Modify Omega_m while keeping the curvature of the Universe the same."""
        self.Omega_m = 1.0 - self.Omega_k - self.Omega_lambda
    
    def Omega_m_h2(self):
        '''calculate the physical matter density parameter'''
        h = self.H0 / 100.0 #km/s/Mpc
        return self.Omega_m * h**2
    
    def __str__(self):
        return f"Cosmology with H0={self.H0}, Omega_m={self.Omega_m}, Omega_lambda={self.Omega_lambda}, Omega_k={self.Omega_k}."
    
    #"__str__" is for returning a string representation of the object.

    
def main():
    H0 = 72.0
    Omega_m = 0.3
    Omega_lambda = 0.72
    c = Cosmology(H0, Omega_m, Omega_lambda)
    #Of course, youcan manually change the parametres to represent different cosmological models!

    num_points = 1000
    z_max = 1.0

    def plot_distance_integrand(cosmology, z_max, num_points):
        """Plot the distance integrand as a function of redshift."""
        z = np.linspace(0, z_max, num_points)
        integrand = cosmology.distance_integrand(z)

        plt.plot(z, integrand, label='Distance Integrand')
        plt.xlabel('Redshift (z)')
        plt.ylabel('Distance Integrand')
        plt.title('Distance Integrand vs Redshift')
        plt.legend()
        plt.show()

    def plot_distance_integrand_with_varying_Omega_m(cosmology, z_max, num_points, Omega_m_values=[0.2, 0.3, 0.4]):
        """Plot the distance integrand for varying Omega_m values."""
        z = np.linspace(0, z_max, num_points)
    
        for Omega_m in Omega_m_values:
            cosmology.Omega_m = Omega_m
            #Without keeping the curvature constant
            integrand = cosmology.distance_integrand(z)
            plt.plot(z, integrand, label=f'Omega_m={Omega_m}, Omega_lambda={cosmology.Omega_lambda:.2f}')

        plt.xlabel('Redshift (z)')
        plt.ylabel('Distance Integrand')
        plt.title('Distance Integrand vs Redshift for Varying Omega_m without fixing curvature')
        plt.legend()
        plt.show()
        
        for Omega_m in Omega_m_values:
            cosmology.Omega_m = Omega_m
            # Adjust Omega_lambda to keep curvature constant
            cosmology.modify_Omega_lambda()  
            integrand = cosmology.distance_integrand(z)
            plt.plot(z, integrand, label=f'Omega_m={Omega_m}, Omega_lambda={cosmology.Omega_lambda:.2f}')

        plt.xlabel('Redshift (z)')
        plt.ylabel('Distance Integrand')
        plt.title('Distance Integrand vs Redshift for Varying Omega_m by fixing curvature')
        plt.legend()
        plt.show()

    plot_distance_integrand(c, z_max, num_points)
    plot_distance_integrand_with_varying_Omega_m(c, z_max, num_points)
    print(c)

# This is a special python idiom that
# allows the code to be run from the command line,
# but if you import this module in another script
# the code below will not be executed.
if __name__ == "__main__":
    main()
