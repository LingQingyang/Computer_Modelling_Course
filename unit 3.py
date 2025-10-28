import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

class Cosmology:
    # Unit 1
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
 
    def whether_flat(self, tol=1e-6):
        """Return True if |Omega_k| < tolerance (Universe is approximately flat)."""
        return abs(self.Omega_k) < tol
    
    def set_Omega_m(self, new_Omega_m):
        """Set Omega_m and adjust Omega_lambda to keep curvature constant."""
        self.Omega_m = new_Omega_m
        self.Omega_lambda = 1.0 - self.Omega_k - self.Omega_m

    def set_Omega_lambda(self, new_Omega_lambda):
        """Set Omega_lambda and adjust Omega_m to keep curvature constant."""
        self.Omega_lambda = new_Omega_lambda
        self.Omega_m = 1.0 - self.Omega_k - self.Omega_lambda
    
    def Omega_m_h2(self):
        '''calculate the physical matter density parameter'''
        h = self.H0 / 100.0 #km/s/Mpc
        return self.Omega_m * h**2
    
    def __str__(self):
        return f"Cosmology with H0={self.H0}, Omega_m={self.Omega_m}, Omega_lambda={self.Omega_lambda}, Omega_k={self.Omega_k}."
    
    #"__str__" is for returning a string representation of the object.

    # Unit 3
    # 4.1.1. Rectangle Rule
    def distance_integral_rectangle(self, z_max, num_points, c):       
        dz = z_max / num_points
        z = np.arange(0, z_max, dz)  # n evenly spaced left endpoints
        f = self.distance_integrand(z)
        integral = np.sum(f) * dz
        return (c / self.H0) * integral

   # 4.1.2. Trapezoid Rule
    def distance_integral_trapezoid(self, z_max, num_points, c):
        z = np.linspace(0, z_max, num_points)
        dz = z[1] - z[0]
        f = self.distance_integrand(z)
        integral = (dz / 2.0) * (f[0] + 2.0 * np.sum(f[1:-1]) + f[-1])
        return (c / self.H0) * integral

    # 4.1.3. Simpson's Rule
    def distance_integral_simpson(self, z_max, num_points, c):
        """Composite Simpson's rule: num_points must be odd (even number of intervals).
        If an even number of points is passed we increment to the next odd value."""
        # Simpson needs an odd number of points
        if num_points % 2 == 0:
            num_points += 1

        z = np.linspace(0, z_max, num_points)
        dz = z[1] - z[0]
        f = self.distance_integrand(z)
        integral = (dz / 3.0) * (
            f[0]
            + 4.0 * np.sum(f[1:-1:2])    # odd indices
            + 2.0 * np.sum(f[2:-1:2])    # even interior indices (correct slice)
            + f[-1]
        )
        return (c / self.H0) * integral

def main():
    H0 = 72.0
    Omega_m = 0.3
    Omega_lambda = 0.7
    base_model = Cosmology(H0, Omega_m, Omega_lambda)

    n = 1000 # Number of points/intervals
    z_max = 1.0
    c = 3.0e5  # km/s
    
    def plot_distance_integrand(base_model, z_max, num_points):
        """Plot the distance integrand as a function of redshift."""
        z = np.linspace(0, z_max, num_points)
        integrand = base_model.distance_integrand(z)

        plt.plot(z, integrand, label='Base Model')
        plt.xlabel('Redshift (z)')
        plt.ylabel('Distance Integrand')
        plt.title('Distance Integrand vs Redshift')
        plt.legend()
        plt.show()

    def plot_distance_integrand_with_varying_Omega_m(base_model, z_max, num_points, Omega_m_values=[0.2, 0.3, 0.4]):
        """Plot the distance integrand for varying Omega_m values."""
        
        for Omega_m in Omega_m_values:
            #Creating new cosmology models and change Omega_m directly
            model = Cosmology(base_model.H0, Omega_m, base_model.Omega_lambda)
            #Plot the shape of the integrand curve by varying Omega_m while fixing Omega_lambda
            z = np.linspace(0, z_max, num_points)
            integrand = model.distance_integrand(z)
            plt.plot(
                z, integrand,
                label=f'Omega_m={Omega_m:.2f}, Omega_lambda={model.Omega_lambda:.2f}, Omega_k={model.Omega_k:.2f}'
            )

        plt.xlabel('Redshift (z)')
        plt.ylabel('Distance Integrand')
        plt.title('Distance Integrand vs Redshift for Varying Omega_m without fixing curvature')
        plt.legend()
        plt.show()
        
        for Omega_m in Omega_m_values:
            #Creating new cosmology models and change Omega_m by setter method
            model = Cosmology(base_model.H0, base_model.Omega_m, base_model.Omega_lambda)
            model.set_Omega_m(Omega_m)
            #Plot the shape of the integrand curve by varying Omega_m while adjusting Omega_lambda to keep curvature constant
            integrand = model.distance_integrand(z)
            plt.plot(
                z, integrand,
                label=f'Omega_m={Omega_m:.2f}, Omega_lambda={model.Omega_lambda:.2f}, Omega_k={model.Omega_k:.2f}'
            )
            
        plt.xlabel('Redshift (z)')
        plt.ylabel('Distance Integrand')
        plt.title('Distance Integrand vs Redshift for Varying Omega_m by fixing curvature')
        plt.legend()
        plt.show()
    
    def convergence_test(base_model, c, z_max=1.0):
        #Explore how the accuracy of each numerical integration method changes with the number of steps n.
            
        # Step 1: Get a very precise 'true' value
        D_true = base_model.distance_integral_simpson(z_max, num_points=10000, c=c)

        # Step 2: Prepare arrays of n values
        n_values = np.logspace(1, 5, 12, dtype=int)

        rect_errors = []
        trap_errors = []
        simp_errors = []

        for n in n_values:
            D_rect = base_model.distance_integral_rectangle(z_max, n, c)
            D_trap = base_model.distance_integral_trapezoid(z_max, n, c)
            D_simp = base_model.distance_integral_simpson(z_max, n, c)

            rect_errors.append(abs(D_rect - D_true) / D_true)
            trap_errors.append(abs(D_trap - D_true) / D_true)
            simp_errors.append(abs(D_simp - D_true) / D_true)

        # Step 3: Plot
        plt.figure(figsize=(8,6))
        plt.loglog(n_values, rect_errors, 'o-', label='Rectangle rule')
        plt.loglog(n_values, trap_errors, 's-', label='Trapezoid rule')
        plt.loglog(n_values, simp_errors, '^-', label="Simpson's rule")

        plt.xlabel('Number of steps n')
        plt.ylabel('Absolute fractional error')
        plt.title('Convergence of Numerical Integration Methods')
        plt.legend()
        plt.show()

        # Step 4: Print results
        print("Final Rectangle result (n=10000):", base_model.distance_integral_rectangle(z_max, 10000, c))
        print("Final Trapezoid result (n=10000):", base_model.distance_integral_trapezoid(z_max, 10000, c))
        print("Final Simpson result (n=10000):", base_model.distance_integral_simpson(z_max, 10000, c))
    
    plot_distance_integrand(base_model, z_max, n)
    plot_distance_integrand_with_varying_Omega_m(base_model, z_max, n)
    print(base_model)
    
    D_rect = base_model.distance_integral_rectangle(z_max, n, c)
    D_trap = base_model.distance_integral_trapezoid(z_max, n, c)
    D_simp = base_model.distance_integral_simpson(z_max, n, c)

    print(f"Rectangle rule: {D_rect:.2f} Mpc")
    print(f"Trapezoid rule: {D_trap:.2f} Mpc")
    print(f"Simpson's rule: {D_simp:.2f} Mpc")

    convergence_test(base_model, c)

# This is a special python idiom that
# allows the code to be run from the command line,
# but if you import this module in another script
# the code below will not be executed.
if __name__ == "__main__":
    main()

