import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

# parâmetros
hbar = 1 # energias medidas em termos de \hbar\omega e tempo em termos de \omega^{-1}
#hbar = 1.05e-34   

omega = 1 # Frequência associada ao Hamiltoniano

# Hamiltoniano 2x2 
H = 0.5 * omega * np.array([[0, 1],
                            [1, 0]])

# Estado inicial 
psi_0 = np.array([1, 0], dtype=complex)

# Tempo de evolução
t_values = np.linspace(0,10, 100)

# Armazenar evolução do estado
prob_0 = []
prob_1 = []

for t in t_values:
    #U_t = np.exp(-1*H*t/hbar)
    U_t = expm(-1j * H * t / hbar)  # Cálculo da exponencial de matriz
    psi_t = U_t @ psi_0             # Evolução do estado
    prob_0.append(abs(psi_t[0])**2) # Probabilidade de estar no estado |0⟩
    prob_1.append(abs(psi_t[1])**2) # Probabilidade de estar no estado |1⟩


# Plotando os resultados
plt.figure(figsize=(8.5, 6))
plt.plot(t_values, prob_0, label=r'$P_0 = |\langle 0 | \psi(t) \rangle|^2$') 
plt.plot(t_values, prob_1, label=r'$P_1 = |\langle 1 | \psi(t) \rangle|^2$', linestyle='dashdot')
plt.xlabel('Tempo (t)')
plt.ylabel('Probabilidade')
plt.title('Evolução Temporal do Estado Quântico')
plt.legend()
plt.grid()
plt.show()
