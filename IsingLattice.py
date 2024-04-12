from typing import Tuple
import numpy as np
kb = 1

class IsingLattice:
    E = 0.0
    E2 = 0.0
    M = 0.0
    M2 = 0.0

    n_steps = 1

    def __init__(self, n_rows, n_cols):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.lattice = np.random.choice([-1, 1], size=(n_rows, n_cols))
        self.J = 1
        self.n_cycles = 30
        self.skip_n_steps = self.n_cycles*self.n_rows*self.n_cols

    def energy(self) -> float:
        """Return the total energy of the current lattice configuration."""
        verticle_pair_sums = np.multiply(self.lattice,np.roll(self.lattice,-1,axis=1))+np.multiply(self.lattice,np.roll(self.lattice,1,axis=1))
        horizontal_pair_sums = np.roll(self.lattice,self.n_rows)*self.lattice + self.lattice*np.roll(self.lattice,self.n_rows)
        return -1/2 *self.J * np.sum(horizontal_pair_sums+verticle_pair_sums)

    def magnetisation(self) -> float:
        """Return the total magnetisation of the current lattice configuration."""
        return np.sum(self.lattice)

    def montecarlostep(self, temp: float) -> Tuple[float, float]:
        """A single Monte-Carlo trial move. Attempts to flip a radnom spin.
        Returns the energy and magnetisation of the new configuration.
        """
        # complete this function so that it performs a single Monte Carlo step
        #energy = self.energy() # comment this line when using delta_energy method
        # the following two lines will select the coordinates of the random spin for you
        random_i = np.random.choice(range(0, self.n_rows))
        random_j = np.random.choice(range(0, self.n_cols))
        self.lattice[random_i][random_j] = -1 * self.lattice[random_i][random_j]
        neg_delta_E = self.delta_energy(random_i,random_j) #finds delta_E faster, directly
        # test against boltzmann if E increase
        if neg_delta_E >= 0:
            random_number = np.random.random() # the following line will choose a random number in the rang e[0,1) for you
            boltzmann_factor = np.e**(-1*neg_delta_E/(kb*temp))
            if random_number > boltzmann_factor: # change lattice back
                self.lattice[random_i][random_j] = -1 * self.lattice[random_i][random_j]
        self.n_steps += 1
        energy = self.energy()
        magnetisation = self.magnetisation()
        if self.n_steps > self.skip_n_steps:
            self.E += energy
            self.E2 += energy**2
            self.M += magnetisation
            self.M2 += magnetisation**2
        return energy, magnetisation

    def statistics(self) -> Tuple[float, float, float, float, int]:
        """Returns the averaged values of energy, energy squared, magnetisation,
        magnetisation squared, and the current step."""
        # complete this function so that it calculates the correct values for the averages of E, E*E (E2), M, M*M (M2), and returns them with Nsteps
        corrected_n_steps = self.n_steps - self.skip_n_steps
        return self.E/corrected_n_steps, self.E2/corrected_n_steps, self.M/corrected_n_steps, self.M2/corrected_n_steps, self.n_steps

    def delta_energy(self, i: int, j: int) -> float:
        """Return the change in energy if the spin at (i,j) were to be flipped."""
        lhs = j-1
        rhs = j+1
        abv = i-1
        blw = i+1
        if i == self.n_rows-1:
            blw = 0
        elif i == 0:
            abv = -1
        if j == self.n_cols-1:
            rhs = 0
        elif j == 0:
            lhs = -1
        interactions = self.lattice[i][j]*self.lattice[abv][j] + self.lattice[i][j]*self.lattice[blw][j] + self.lattice[i][j]*self.lattice[i][lhs] + self.lattice[i][j]*self.lattice[i][rhs]
        return -2*self.J*interactions
