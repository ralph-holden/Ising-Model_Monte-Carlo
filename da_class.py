# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 16:58:06 2024

@author: Ralph Holden
"""
import numpy as np
import matplotlib.pyplot as plt


class Figure:
    
    beta = 1
    kb = 1
    temperature_label = 'Temperature ($J\:k_b^{-1}$)'
    heat_capacity_label = 'Heat Capacity per Spin ($k_b\:J^{-1}\:N^{-3}$)'
    energy_label = 'Mean Energy (J)'
    mag_label = 'Mean Magnetisation'
    mag_sus_label = 'Magnetic Susceptability ($J^{-1}$)'
    
    def __init__(self, filename_Cpp, filename):
        self.filename = filename
        self.data = np.loadtxt(filename+".dat",delimiter=' ')
 
        self.factor = float(filename[0])**-4
        self.n_spins = float(filename[0])**2       
 
        self.temp = self.data[:,0]
        self.E = self.data[:,1]#*self.factor
        self.E2 = self.data[:,2]#/self.n_spins
        self.M = self.data[:,3]#*self.factor
        self.M2 = self.data[:,4]#/self.n_spins
        self.Var_E = self.E2 - self.E**2
        self.Var_M = self.M2 - self.M**2
    
        
        self.data_Cpp = np.loadtxt("./C++_data/"+filename_Cpp+".dat",delimiter=' ')
        self.temp_Cpp = self.data_Cpp[:,0]
        self.E_Cpp = self.data_Cpp[:,1]
        self.E2_Cpp = self.data_Cpp[:,2]
        self.M_Cpp = self.data_Cpp[:,3]
        self.M2_Cpp = self.data_Cpp[:,4]
        self.C_Cpp = self.data_Cpp[:,5]
        self.Var_E_Cpp = self.E2_Cpp - self.E_Cpp**2
        #self.Var_E_Cpp = ((self.E2_Cpp*self.n_spins) - (self.E_Cpp*self.n_spins)**2)/self.n_spins
        self.Var_M_Cpp = self.M2_Cpp - self.M_Cpp**2

    def plot_5(self):
        
        plt.figure(figsize=[8,10])
        plt.subplot(2,1,1)
        plt.title('Average Energy as a Function of Temperature')
        plt.xlabel(self.temperature_label)
        plt.ylabel(self.energy_label)
        plt.grid(linestyle=':')
        plt.plot(self.temp,self.E,color='blue')
        
        plt.subplot(2,1,2)
        plt.title('Average Magnetisation as a Function of Temperature')
        plt.xlabel(self.temperature_label)
        plt.ylabel(self.mag_label)
        plt.grid(linestyle=':')
        plt.plot(self.temp,self.M,color='red')
        
        plt.tight_layout(pad=2)
    
    def plot_7(self):
        plt.figure(figsize=[8,10])
        plt.subplot(2, 1, 1)
        plt.title('Heat Capacity as a Function of Temperature')
        plt.xlabel(self.temperature_label)
        plt.ylabel(self.heat_capacity_label)
        plt.grid(linestyle=':')
        plt.plot(self.temp,(1/(self.kb*self.temp**2))*self.Var_E*self.factor,color='blue',label='Python')

        plt.subplot(2, 1, 2)
        plt.title('Magnetic Susceptability as a Function of Temperature')
        plt.xlabel(self.temperature_label)
        plt.ylabel(self.mag_sus_label)
        plt.grid(linestyle=':')
        plt.plot(self.temp1,1/self.temp*self.Var_M*self.factor,color='red',label='Python')
        
        plt.tight_layout(pad=2)
        
        plt.savefig('./figures/task7_'+self.filename+'.png')
    
    def plot_8(self):
        plt.figure(figsize=[8,10])
        plt.subplot(2, 1, 1)
        plt.title('Heat Capacity as a Function of Temperature')
        plt.xlabel(self.temperature_label)
        plt.ylabel(self.heat_capacity_label)
        plt.grid(linestyle=':')
        plt.plot(self.temp,(1/self.temp**2)*self.Var_E/self.n_spins,color='blue',label='Python')
        plt.plot(self.temp_Cpp,(1/(self.kb*self.temp_Cpp**2))*self.C_Cpp,color='blue',linestyle=':',label='C++')
        plt.legend(loc='best')

        plt.subplot(2, 1, 2)
        plt.title('Magnetic Susceptability as a Function of Temperature')
        plt.xlabel(self.temperature_label)
        plt.ylabel(self.mag_sus_label)
        plt.grid(linestyle=':')
        plt.plot(self.temp,(1/self.temp)*self.Var_M/self.n_spins,color='red',label='Python')
        plt.plot(self.temp_Cpp,(1/self.temp_Cpp)*self.Var_M_Cpp*self.n_spins,color='red',linestyle=':',label='C++')
        plt.legend(loc='best')
        
        plt.tight_layout(pad=2)
        
        plt.savefig('./figures/task8_'+self.filename+'.png')
        
    def plot_8_corr(self,FACTOR):

        plt.figure(figsize=[8,10])
        plt.subplot(2, 1, 1)
        plt.title('Heat Capacity as a Function of Temperature')
        plt.xlabel(self.temperature_label)
        plt.ylabel(self.heat_capacity_label)
        plt.grid(linestyle=':')
        plt.plot(self.temp,(1/self.temp**2)*self.Var_E/FACTOR**2,color='blue',label='Python')
        plt.plot(self.temp_Cpp,(1/(self.kb*self.temp_Cpp**2))*self.Var_E_Cpp,color='blue',linestyle=':',label='C++')
        plt.legend(loc='best')

        plt.subplot(2, 1, 2)
        plt.title('Magnetic Susceptability as a Function of Temperature')
        plt.xlabel(self.temperature_label)
        plt.ylabel(self.mag_sus_label)
        plt.grid(linestyle=':')
        plt.plot(self.temp,(1/self.temp)*self.Var_M/FACTOR**2,color='red',label='Python')
        plt.plot(self.temp_Cpp,(1/self.temp_Cpp)*self.Var_M_Cpp,color='red',linestyle=':',label='C++')
        plt.legend(loc='best')

        plt.tight_layout(pad=2)

        plt.savefig('./figures/task8_'+self.filename+'.png')
        
    def plot_Cpp(self):
        plt.figure(figsize=[8,5])
        plt.title('Heat Capacity as a Function of Temperature')
        plt.xlabel(self.temperature_label)
        plt.ylabel(self.heat_capacity_label)
        plt.grid(linestyle=':')
        plt.plot(self.temp_Cpp,(1/(self.kb*self.temp_Cpp**2))*self.Var_E_Cpp,color='blue',linestyle=':',label='C++')
        plt.legend(loc='best')

        
    def plot_curie_all(self):
        T_min = np.min(self.temp_Cpp)
        T_max = np.max(self.temp_Cpp)
        T_range = np.linspace(T_min, T_max, 1000) #generate 1000 evenly spaced points between T_min and T_max
        
        C_values = (1/(self.kb*self.temp**2))*self.Var_E*self.factor
        M_values = (1/(self.kb*self.temp))*self.Var_M*self.factor
        
        fit = np.polyfit(peak_T_values, C_values, 3) # fit a third order polynomial
        fitted_C_values = np.polyval(fit, T_range) # use the fit object to generate the corresponding values of C
        
        fit_M = np.polyfit(peak_T_values, M_values, 4) # fit a third order polynomial
        fitted_M_values = np.polyval(fit_M, T_range) # use the fit object to generate the corresponding values of C        
        
        plt.figure(figsize=[8,10])
        plt.subplot(2, 1, 1)
        plt.title('Heat Capacity as a Function of Temperature')
        plt.xlabel('Temperature')
        plt.ylabel(self.heat_capacity_label)
        plt.grid(linestyle=':')
        #plt.plot(self.temp,(1/self.temp**2)*self.Var_E/self.factor,color='blue',label='Python')
        plt.plot(self.temp_Cpp,(1/(self.kb*self.temp_Cpp**2))*self.Var_E_Cpp,color='blue',marker='.',linestyle='',label='C++')
        plt.plot(T_range,fitted_C_values,color='black',marker='',linestyle='-',label='Polynomial Fit')
        plt.legend(loc='best')

        plt.subplot(2, 1, 2)
        plt.title('Magnetic Susceptability as a Function of Temperature')
        plt.xlabel('Temperature')
        plt.ylabel('Magnetic Susceptability per Spin')
        plt.grid(linestyle=':')
        plt.plot(self.temp_Cpp,(self.temp_Cpp)*self.Var_M_Cpp,color='red',linestyle=':',label='C++')
        plt.plot(T_range,fitted_M_values,color='black',marker='',linestyle='-',label='Polynomial Fit')
        plt.legend(loc='best')
        
        plt.tight_layout(pad=2)
        
        plt.savefig('./figures/task8b_'+self.filename+'.png')
        
    def plot_curie(self, Tmin, Tmax, TMmin, TMmax):
        selection = np.logical_and(self.temp_Cpp > Tmin, self.temp_Cpp < Tmax) 
        peak_T_values = self.temp_Cpp[selection]
        peak_C_values = self.C_Cpp[selection]
       
        fit = np.polyfit(peak_T_values, peak_C_values, 3)
        fitted_C_values = np.polyval(fit, peak_T_values)
        Cmax = np.max(fitted_C_values)
        Tmax = peak_T_values[fitted_C_values == Cmax]
        
        selection_M = np.logical_and(self.temp_Cpp > TMmin, self.temp_Cpp < TMmax) 
        peak_TM_values = self.temp_Cpp[selection_M]
        peak_M_values = (1/self.temp_Cpp*self.Var_M_Cpp)[selection_M]
        
        fit = np.polyfit(peak_TM_values, peak_M_values, 4)
        fitted_M_values = np.polyval(fit, peak_TM_values)
        Mmax = np.max(fitted_M_values)
        TMmax = peak_TM_values[fitted_M_values == Mmax]
        
        plt.figure(figsize=[8,10])
        plt.subplot(2, 1, 1)
        plt.title('Heat Capacity as a Function of Temperature')
        plt.xlabel(self.temperature_label)
        plt.ylabel(self.heat_capacity_label)
        plt.grid(linestyle=':')
        plt.plot(self.temp_Cpp,self.C_Cpp,color='blue',marker='.',linestyle='',label='C++')
        plt.plot(peak_T_values,fitted_C_values,color='black',marker='',linestyle='-',linewidth=3,label='Polynomial Fit')
        plt.legend(loc='best')

        plt.subplot(2, 1, 2)
        plt.title('Magnetic Susceptability as a Function of Temperature')
        plt.xlabel(self.temperature_label)
        plt.ylabel(self.mag_sus_label)
        plt.grid(linestyle=':')
        plt.plot(self.temp_Cpp,(1/self.temp_Cpp)*self.Var_M_Cpp,color='red',linestyle=':',label='C++')
        plt.plot(peak_TM_values,fitted_M_values,color='black',marker='',linestyle='-',linewidth=3,label='Polynomial Fit')
        plt.legend(loc='best')
        
        plt.tight_layout(pad=2)
        
        plt.savefig('./figures/task8c_'+self.filename+'.png')
        
        return Tmax, Cmax
    
    def plot_curie_mine(self, Tmin, Tmax):
        selection = np.logical_and(self.temp > Tmin, self.temp < Tmax) 
        peak_T_values = self.temp[selection]
        C_values = (1/(self.kb*self.temp**2))*self.Var_E*self.factor
        peak_C_values = C_values[selection]
        
        fit, cov = np.polyfit(peak_T_values, peak_C_values, 3, cov=True)
        fitted_C_values = np.polyval(fit, peak_T_values)
        Cmax = np.max(fitted_C_values)
        Tmax = peak_T_values[fitted_C_values == Cmax]
        
        plt.figure(figsize=[8,5])
        #plt.subplot(2, 1, 1)
        plt.title('Heat Capacity as a Function of Temperature')
        plt.xlabel(self.temperature_label)
        plt.ylabel(self.heat_capacity_label)
        plt.grid(linestyle=':')
        plt.plot(self.temp,C_values,color='blue',marker='.',linestyle='',label='Python')
        plt.plot(peak_T_values,fitted_C_values,color='black',marker='',linestyle='-',linewidth=3,label='Polynomial Fit')
        plt.legend(loc='best')
        
        plt.savefig('./figures/task8c_'+self.filename+'.png')
        
        return Tmax, Cmax
    
    def plot_curie_M(self, Tmin, Tmax):
        selection = np.logical_and(self.temp_Cpp > Tmin, self.temp_Cpp < Tmax) 
        peak_T_values = self.temp_Cpp[selection]
        peak_M_values = (1/self.temp_Cpp*self.Var_M_Cpp)[selection]
        
        fit = np.polyfit(peak_T_values, peak_M_values, 3)
        fitted_M_values = np.polyval(fit, peak_T_values)
        Mmax = np.max(fitted_M_values)
        Tmax = peak_T_values[fitted_M_values == Mmax]
        
        plt.figure(figsize=[8,5])
        #plt.subplot(2, 1, 1)
        plt.title('Magnetic Susceptability as a Function of Temperature')
        plt.xlabel(self.temperature_label)
        plt.ylabel(self.mag_sus_label)
        plt.grid(linestyle=':')
        plt.plot(self.temp_Cpp,(1/self.temp_Cpp)*self.Var_M_Cpp,color='red',linestyle=':',label='C++')
        plt.plot(peak_T_values,fitted_M_values,color='black',marker='',linestyle='-',linewidth=3,label='Polynomial Fit')
        plt.legend(loc='best')
        
        plt.savefig('./figures/task8c_M_'+self.filename+'.png')
        
        return Tmax, Mmax

class Figure2:
    
    beta = 1
    kb = 1
    temperature_label = 'Temperature (J $k_b^{-1}$)'
    heat_capacity_label = 'Heat Capacity per Spin ($k_b J^{-1} N^{-3}$)'
    energy_label = 'Mean Energy per Spin (J)'
    mag_label = 'Mean Magnetisation per Spin'
    mag_sus_label = 'Magnetic Susceptability $J^{-1}$'
    
    def __init__(self,filename,file_list,n=5):

        self.data0 = np.loadtxt(file_list[0]+".dat",delimiter=' ')
        self.data1 = np.loadtxt(file_list[1]+".dat",delimiter=' ')
        self.data2 = np.loadtxt(file_list[2]+".dat",delimiter=' ')
        self.data3 = np.loadtxt(file_list[3]+".dat",delimiter=' ')
        self.data4 = np.loadtxt(file_list[4]+".dat",delimiter=' ')
        self.temp = self.data0[:,0]
        self.E = np.mean([self.data0[:,1],self.data1[:,1],self.data2[:,1],self.data3[:,1],self.data4[:,1]],axis=0)
        self.E_std = np.std([self.data0[:,1],self.data1[:,1],self.data2[:,1],self.data3[:,1],self.data4[:,1]],axis=0)
        self.E2 = np.mean([self.data0[:,2],self.data1[:,2],self.data2[:,2],self.data3[:,2],self.data4[:,2]],axis=0)
        self.E2_std = np.std([self.data0[:,2],self.data1[:,2],self.data2[:,2],self.data3[:,2],self.data4[:,2]])
        self.M = np.mean([self.data0[:,3],self.data1[:,3],self.data2[:,3],self.data3[:,3],self.data4[:,3]],axis=0)
        self.M_std = np.std([self.data0[:,3],self.data1[:,3],self.data2[:,3],self.data3[:,3],self.data4[:,3]],axis=0)
        self.M2 = np.mean([self.data0[:,4],self.data1[:,4],self.data2[:,4],self.data3[:,4],self.data4[:,4]],axis=0)
        self.M2_std = np.std([self.data0[:,4],self.data1[:,4],self.data2[:,4],self.data3[:,4],self.data4[:,4]],axis=0)
        self.Var_E = self.E2 - self.E**2
        self.Var_E_std = np.std(self.Var_E) #self.E2_std + 2*self.E_std
        self.Var_M = self.M2 - self.M**2
        self.Var_M_std = self.M2_std + 2*self.M_std
        
        self.factor = float(filename[0])**-4
        self.n_spins = float(filename[0])**2
        
        self.filename = filename
        self.data_Cpp = np.loadtxt("./C++_data/"+filename+".dat",delimiter=' ')
        self.temp_Cpp = self.data_Cpp[:,0]
        self.E_Cpp = self.data_Cpp[:,1]
        self.E2_Cpp = self.data_Cpp[:,2]
        self.M_Cpp = self.data_Cpp[:,3]
        self.M2_Cpp = self.data_Cpp[:,4]
        self.C_Cpp = self.data_Cpp[:,5]
        self.Var_E_Cpp = self.E2_Cpp - self.E_Cpp**2
        #self.Var_E_Cpp = ((self.E2_Cpp*self.n_spins) - (self.E_Cpp*self.n_spins)**2)/self.n_spins
        self.Var_M_Cpp = self.M2_Cpp - self.M_Cpp**2
   
    
    def plot_5(self):
        
        plt.figure(figsize=[8,10])
        plt.subplot(2,1,1)
        plt.title('Average Energy as a Function of Temperature')
        plt.xlabel(self.temperature_label)
        plt.ylabel(self.energy_label)
        plt.grid(linestyle=':')
        plt.errorbar(self.temp,self.E/self.n_spins,yerr=self.E_std/self.n_spins,color='blue',capsize=3)
        
        plt.subplot(2,1,2)
        plt.title('Average Magnetisation as a Function of Temperature')
        plt.xlabel(self.temperature_label)
        plt.ylabel(self.mag_label)
        plt.grid(linestyle=':')
        plt.errorbar(self.temp,self.M/self.n_spins,yerr=self.M_std/self.n_spins,color='red',capsize=3)
        
        plt.tight_layout(pad=2)
        
        plt.savefig('./figures/task6_'+self.filename+'.png')
    
    def plot_7(self):
        plt.figure(figsize=[8,10])
        plt.subplot(2, 1, 1)
        plt.title('Heat Capacity as a Function of Temperature')
        plt.xlabel(self.temperature_label)
        plt.ylabel(self.heat_capacity_label)
        plt.grid(linestyle=':')
        plt.plot(self.temp,(1/(self.kb*self.temp**2))*self.Var_E*self.factor,color='blue',label='Python')
        #plt.errorbar(self.temp,(1/(self.kb*self.temp**2))*self.Var_E*self.factor,yerr=self.Var_E_std,color='blue')

        plt.subplot(2, 1, 2)
        plt.title('Magnetic Susceptability as a Function of Temperature')
        plt.xlabel(self.temperature_label)
        plt.ylabel(self.mag_sus_label)
        plt.grid(linestyle=':')
        plt.plot(self.temp,(self.temp)*self.Var_M*self.factor,color='red',label='Python')
        #plt.errorbar(self.temp,(self.temp)*self.Var_M*self.factor,yerr=self.Var_M_std,color='red')
        
        plt.tight_layout(pad=2)
        
        plt.savefig('./figures/task7_'+self.filename+'.png')
    
    def plot_8(self):
        plt.figure(figsize=[8,10])
        plt.subplot(2, 1, 1)
        plt.title('Heat Capacity as a Function of Temperature')
        plt.xlabel(self.temperature_label)
        plt.ylabel(self.heat_capacity_label)
        plt.grid(linestyle=':')
        plt.plot(self.temp,(1/self.temp**2)*self.Var_E*self.factor**2,color='blue',label='Python')
        plt.plot(self.temp_Cpp,(1/(self.kb*self.temp_Cpp**2))*self.Var_E_Cpp,color='blue',linestyle=':',label='C++')
        plt.legend(loc='best')

        plt.subplot(2, 1, 2)
        plt.title('Magnetic Susceptability as a Function of Temperature')
        plt.xlabel(self.temperature_label)
        plt.ylabel(self.mag_sus_label)
        plt.grid(linestyle=':')
        plt.plot(self.temp,(self.temp)*self.Var_M*self.factor**2,color='red',label='Python')
        plt.plot(self.temp_Cpp,(self.temp_Cpp)*self.Var_M_Cpp,color='red',linestyle=':',label='C++')
        plt.legend(loc='best')
        
        plt.tight_layout(pad=2)
        
        plt.savefig('./figures/task8_'+self.filename+'.png')
        
    def plot_8_corr(self,FACTOR):

        plt.figure(figsize=[8,10])
        plt.subplot(2, 1, 1)
        plt.title('Heat Capacity as a Function of Temperature')
        plt.xlabel(self.temperature_label)
        plt.ylabel(self.heat_capacity_label)
        plt.grid(linestyle=':')
        plt.plot(self.temp,(1/self.temp**2)*self.Var_E/FACTOR**2,color='blue',label='Python')
        plt.plot(self.temp_Cpp,(1/(self.kb*self.temp_Cpp**2))*self.Var_E_Cpp,color='blue',linestyle=':',label='C++')
        plt.legend(loc='best')

        plt.subplot(2, 1, 2)
        plt.title('Magnetic Susceptability as a Function of Temperature')
        plt.xlabel(self.temperature_label)
        plt.ylabel(self.mag_sus_label)
        plt.grid(linestyle=':')
        plt.plot(self.temp,(self.temp)*self.Var_M/FACTOR**2,color='red',label='Python')
        plt.plot(self.temp_Cpp,(self.temp_Cpp)*self.Var_M_Cpp,color='red',linestyle=':',label='C++')
        plt.legend(loc='best')

        plt.tight_layout(pad=2)

        plt.savefig('./figures/task8_'+self.filename+'.png')
        
    def plot_curie(self, Tmin, Tmax):
        selection = np.logical_and(self.temp > Tmin, self.temp < Tmax) 
        peak_T_values = self.temp[selection]
        C_values = (1/(self.kb*self.temp**2))*self.Var_E*self.factor
        C_err = (1/(self.kb*self.temp**2))*self.Var_E_std*self.factor # (1/(self.kb*self.temp**2))*
        peak_C_values = C_values[selection]
        
        fit, cov = np.polyfit(peak_T_values, peak_C_values, 3, cov=True)
        fitted_C_values = np.polyval(fit, peak_T_values)
        Cmax = np.max(fitted_C_values)
        Tmax = peak_T_values[fitted_C_values == Cmax]
        
        C_err = (cov[0,0])**0.5*(Tmax-Tmin)*np.ones(len(C_values))
        
        plt.figure(figsize=[8,5])
        #plt.subplot(2, 1, 1)
        plt.title('Heat Capacity as a Function of Temperature')
        plt.xlabel(self.temperature_label)
        plt.ylabel(self.heat_capacity_label)
        plt.grid(linestyle=':')
        plt.errorbar(self.temp,C_values,yerr=C_err,color='blue',marker='.',linestyle='',label='Python')
        plt.plot(peak_T_values,fitted_C_values,color='black',marker='',linestyle='-',linewidth=3,label='Polynomial Fit')
        plt.legend(loc='best')
        
        plt.savefig('./figures/task8c_'+self.filename+'.png')
        
        return Tmax, Cmax, C_err