#coding:utf8

import numpy as np
import pandas as pd
import scipy
import scipy.stats

#https://docs.scipy.org/doc/scipy/reference/stats.html


dist_names = ['norm', 'beta', 'gamma', 'pareto', 't', 'lognorm', 'invgamma', 'invgauss',  'loggamma', 'alpha', 'chi', 'chi2', 'bradford', 'burr', 'burr12', 'cauchy', 'dweibull', 'erlang', 'expon', 'exponnorm', 'exponweib', 'exponpow', 'f', 'genpareto', 'gausshyper', 'gibrat', 'gompertz', 'gumbel_r', 'pareto', 'pearson3', 'powerlaw', 'triang', 'weibull_min', 'weibull_max', 'bernoulli', 'betabinom', 'betanbinom', 'binom', 'geom', 'hypergeom', 'logser', 'nbinom', 'poisson', 'poisson_binom', 'randint', 'zipf', 'zipfian']

print(dist_names)



# QUESTION 1
import os
import matplotlib.pyplot as plt
os.makedirs("graphiques_visualisation_lois", exist_ok=True)

#Loi de Dirac: 
a = 2
k_dirac = np.array([a])         
probabilité_dirac = np.array([1.0])   
plt.figure()    
plt.bar(k_dirac, probabilité_dirac, width=0.1, color='blue')
plt.title("Loi de Dirac centrée en a = {}".format(a))
plt.xlabel("Valeur")
plt.ylabel("Probabilité")
plt.ylim(0, 1.2)
plt.savefig("graphiques_visualisation_lois/loi_dirac.png", bbox_inches='tight')

#Loi uniforme discrète : 
a = 1
b = 10
k_uniforme = np.arange(a, b + 1)
probabilité_uniforme = np.ones_like(k_uniforme) / len(k_uniforme)
plt.figure()
plt.bar(k_uniforme
, probabilité_uniforme, color="green")
plt.title(f"Loi uniforme discrète de {a} à {b}")
plt.xlabel("Valeurs")
plt.ylabel("Probabilité")
plt.ylim(0, max(probabilité_uniforme)* 1.2)
plt.savefig("graphiques_visualisation_lois/loi_uniforme_discrete.png", bbox_inches="tight")


#Loi binomiale : 
from scipy.stats import binom
n = 10      
p = 0.5
k_binomiale = np.arange(0, n + 1)
probabilité_binom = binom.pmf(k_binomiale, n, p)
plt.figure()
plt.bar(k_binomiale, probabilité_binom, color="blue")
plt.title(f"Loi binomiale(n={n}, p={p})")
plt.xlabel("Valeurs")
plt.ylabel("Probabilité P(X=k)")
plt.savefig("graphiques_visualisation_lois/loi_binomiale.png", bbox_inches="tight")


#Loi de Poisson :
from scipy.stats import poisson
lam = 2
k_poisson = np.arange(0, 15)
probabilité_poisson = poisson.pmf(k_poisson, mu=lam)
plt.figure()
plt.bar(k_poisson, probabilité_poisson, color="pink")
plt.title(f"Loi de Poisson (λ={lam})")
plt.xlabel("Nombre d'événements k")
plt.ylabel("Probabilité P(X=k)")
plt.savefig("graphiques_visualisation_lois/Loi_de_Poisson.png", bbox_inches="tight")


#Loi de Zipf-Mandelbrot
N = 50     
q = 1.0    
s = 1.0   
k_zipf_M = np.arange(1, N+1)
H_N_q_s = np.sum(1.0 / (k_zipf_M + q)**s)
probabilité_Zipf_M = (1.0 / (k_zipf_M + q)**s) / H_N_q_s
plt.figure()
plt.bar(k_zipf_M, probabilité_Zipf_M, color="deeppink")
plt.title(f"Loi de Zipf–Mandelbrot (N={N}, q={q}, s={s})")
plt.xlabel("Valeurs")
plt.ylabel("Probabilité P(X=k)")
plt.savefig("graphiques_visualisation_lois/Loi_de_Zipf-Mandelbrot.png", bbox_inches="tight")


#Loi de Poisson continue : 
from scipy.interpolate import make_interp_spline
lam = 5
k_poisson_cont=np.arange(0, 20)
probabilité_poisson_cont = poisson.pmf(k_poisson_cont, mu=lam)
x_cont = np.linspace(k_poisson_cont.min(), k_poisson_cont.max(), 300)
spl = make_interp_spline(k_poisson_cont, probabilité_poisson_cont, k=3)
y_cont = spl(x_cont)
plt.figure()
plt.plot(x_cont, y_cont, color="#F54927", linewidth=2)
plt.title(f"Loi de Poisson continue (λ={lam})")
plt.xlabel("Valeurs (approximation continue)")
plt.ylabel("Probabilité")
plt.savefig("graphiques_visualisation_lois/Loi_de_Poisson_continue.png", bbox_inches="tight")


#Loi Normale : 
from scipy.stats import norm
mu = 0      
sigma = 1   
k_normale = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)
pdf = norm.pdf(k_normale, loc=mu, scale=sigma)
plt.figure()
plt.plot(k_normale, pdf, color="#F54927", linewidth=2)
plt.title(f"Loi normale continue (μ={mu}, σ={sigma})")
plt.xlabel("Valeurs")
plt.ylabel("Densité de probabilité")
plt.savefig("graphiques_visualisation_lois/Loi_normale_continue.png", bbox_inches="tight")


#Loi Log Normale : 
from scipy.stats import lognorm
mu = 0.0  
sigma = 1.0
k_log = np.linspace(0.01, 5, 500)
pdf_lognorm = lognorm.pdf(k_log, s=sigma, scale=np.exp(mu))
plt.figure()
plt.plot(k_log, pdf_lognorm, color="#F54927", linewidth=2)
plt.title(f"Loi log normale (μ={mu}, σ={sigma})")
plt.xlabel("Valeurs")
plt.ylabel("Densité de probabilité")
plt.savefig("graphiques_visualisation_lois/loi_log_normale.png", bbox_inches="tight")


#Loi uniforme continue :
from scipy.stats import uniform
a =-5
b = 20    
k_uniforme_cont = np.linspace(a - 1, b + 1, 500)
pdf = uniform.pdf(k_uniforme_cont, loc=a, scale=b - a)
plt.figure()
plt.plot(k_uniforme_cont, pdf, color="#F54927", linewidth=2)
plt.title(f"Loi uniforme continue U({a},{b})")
plt.xlabel("Valeurs")
plt.ylabel("Densité de probabilité")
plt.savefig("graphiques_visualisation_lois/Loi_uniforme_continue.png", bbox_inches="tight")


#Loi du Chi²
from scipy.stats import chi2
df = 2
k_chi2 = np.linspace(0, 15, 500)
pdf_chi2 = chi2.pdf(k_chi2, df)
plt.figure()
plt.plot(k_chi2, pdf_chi2, color="#F54927", linewidth=2)
plt.title(f"Loi du Chi²(χ²)(df={df})")
plt.xlabel("Valeurs")
plt.ylabel("Densité de probabilité")
plt.savefig("graphiques_visualisation_lois/Loi_chi2.png", bbox_inches="tight")



# Loi de Pareto:
from scipy.stats import pareto
alpha = 3.0    
x_m = 1.0  
k_pareto = np.linspace(x_m, 10, 500)
pdf_pareto = pareto.pdf(k_pareto/x_m, alpha) / x_m
plt.figure()
plt.plot(k_pareto, pdf_pareto, color="#F54927", linewidth=2)
plt.title(f"Loi de Pareto (α={alpha}, x_m={x_m})")
plt.xlabel("Valeurs")
plt.ylabel("Densité de probabilité")
plt.savefig("graphiques_visualisation_lois/Loi_de_Pareto.png", bbox_inches="tight")





#Question 2 :

# Moyennes et écarts-types
def calcul_moyenne(x, probabilités):
    """
    x : valeurs de la variable aléatoire
    probabilités : probabilités associées à chaque valeur (ou densité pour loi continue)
    """
    # Vérification si la somme des probabilités est proche de 1
    if not np.isclose(np.sum(probabilités), 1):
        # On normalise si nécessaire (utile pour les PDF continues discrétisées)
        probabilités= probabilités / np.sum(probabilités)
    moyenne = np.sum(x * probabilités)
    return moyenne


def calcul_ecart_type(x, probabilités):
    """
    x : valeurs de la variable aléatoire
    probabilités : probabilités associées à chaque valeur (ou densité pour loi continue)
    """
    if not np.isclose(np.sum(probabilités), 1):
        probabilités = probabilités / np.sum(probabilités)
    moyenne = calcul_moyenne(x, probabilités)
    variance = np.sum(probabilités * (x - moyenne)**2)
    return np.sqrt(variance)



moy_dirac = calcul_moyenne(k_dirac,probabilité_dirac)
ecart_dirac = calcul_ecart_type(k_dirac,probabilité_dirac)
print("Moyenne loi de Dirac =",moy_dirac, "\nécart-type =",ecart_dirac)
print("--------")

moy_uniforme = calcul_moyenne(k_uniforme, probabilité_uniforme)
ecart_uniforme = calcul_ecart_type(k_uniforme, probabilité_uniforme)
print("Moyenne loi uniforme = ",moy_uniforme ,"\nécart-type", ecart_uniforme)
print("---------")

moy_binomiale = calcul_moyenne(k_binomiale, probabilité_binom)
ecart_binomiale = calcul_ecart_type(k_binomiale,probabilité_binom)
print(" Moyenne loi binomiale =", moy_binomiale, "\nEcart-type =" , ecart_binomiale)
print("---------")

moy_poisson = calcul_moyenne(k_poisson,probabilité_poisson)
ecart_poisson = calcul_ecart_type(k_poisson,probabilité_poisson)
print("Moyenne loi poisson =", moy_poisson, "\nEcart-type =", ecart_poisson)
print("---------")

moy_zipf_m = calcul_moyenne(k_zipf_M,probabilité_Zipf_M)
ecart_zipf_m = calcul_ecart_type(k_zipf_M,probabilité_Zipf_M)
print("Moyenne loi Zipf-Mandelbrot = " , moy_zipf_m, "\n Ecart-type =", ecart_zipf_m)
print("---------")

moy_poisson_cont = calcul_moyenne(k_poisson_cont,probabilité_poisson_cont)
ecart_poisson_cont = calcul_ecart_type(k_poisson_cont,probabilité_poisson_cont)
print("Moyenne loi Poisson continue =" , moy_poisson_cont, "\nEcart-type =",ecart_poisson_cont)
print("----------")

pdf_normale = norm.pdf(k_normale, loc=mu, scale=sigma)
dx_norm = k_normale[1] - k_normale[0]
probabilité_normale = pdf_normale * dx_norm
moy_normale = calcul_moyenne(k_normale, probabilité_normale)
ecart_normale = calcul_ecart_type(k_normale, probabilité_normale)
print("Moyenne loi normale =", moy_normale, "\nEcart-type =", ecart_normale)
print("---------")

dx_log = k_log[1] - k_log[0]
probabilité_lognorm = pdf_lognorm * dx_log
moy_lognorm = calcul_moyenne(k_log, probabilité_lognorm)
ecart_lognorm = calcul_ecart_type(k_log, probabilité_lognorm)
print("Moyenne loi log-normale =", moy_lognorm, "\nEcart-type =", ecart_lognorm)
print("---------")

dx_unif = k_uniforme_cont[1] - k_uniforme_cont[0]
probabilité_uniforme_cont = pdf * dx_unif
moy_uniforme_cont = calcul_moyenne(k_uniforme_cont, probabilité_uniforme_cont)
ecart_uniforme_cont = calcul_ecart_type(k_uniforme_cont, probabilité_uniforme_cont)
print("Moyenne loi uniforme continue =", moy_uniforme_cont, "\nEcart-type =", ecart_uniforme_cont)
print("---------")

dx_chi2 = k_chi2[1] - k_chi2[0]
probabilité_chi2 = pdf_chi2 * dx_chi2
moy_chi2 = calcul_moyenne(k_chi2, probabilité_chi2)
ecart_chi2 = calcul_ecart_type(k_chi2, probabilité_chi2)
print("Moyenne loi Chi² =", moy_chi2, "\nEcart-type =", ecart_chi2)
print("---------")

dx_pareto = k_pareto[1] - k_pareto[0]
probabilité_pareto = pdf_pareto * dx_pareto
moy_pareto = calcul_moyenne(k_pareto, probabilité_pareto)
ecart_pareto = calcul_ecart_type(k_pareto, probabilité_pareto)
print("Moyenne loi de Pareto =", moy_pareto, "\nEcart-type =", ecart_pareto)
print("---------")







