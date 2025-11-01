import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

try:
    df_seq = pd.read_csv('resultados_seq.csv')
    df_par = pd.read_csv('resultados_mpi.csv')
except FileNotFoundError as e:
    print(f"Erro: Arquivo não encontrado. Certifique-se que o arquivo '{e.filename}' está na mesma pasta que o script.")
    exit()

# Média para a execução sequencial
seq_avg = df_seq.groupby('Tamanho')['Tempo_s'].mean().reset_index()
seq_avg = seq_avg.rename(columns={'Tempo_s': 'Sequencial'})

# Média para a execução paralela (MPI)
par_avg = df_par.groupby(['Tamanho', 'Processos'])['Tempo_s'].mean().reset_index()

print("Médias de tempo de execução (em segundos):")
print("------------------------------------------")
print("Versão Sequencial:")
print(seq_avg)
print("\nVersão Paralela (MPI):")
print(par_avg)
print("------------------------------------------\n")

# Preparar dados para os Gráficos
par_pivot = par_avg.pivot(index='Tamanho', columns='Processos', values='Tempo_s')
proc_counts = sorted(par_pivot.columns.tolist())
df_merged = pd.merge(seq_avg, par_pivot, on='Tamanho')
print("Tabela de dados combinados para os gráficos:")
print(df_merged)
print("------------------------------------------\n")

# Gera o Gráfico de Barras (Tempo de Execução) 

tamanhos = df_merged['Tamanho'].values
x = np.arange(len(tamanhos))
total_bars = 1 + len(proc_counts)
width = 0.8 / total_bars
fig, ax = plt.subplots(figsize=(14, 8))
rects_all = []
offsets = [(i - (total_bars - 1) / 2) * width for i in range(total_bars)]
rects_seq = ax.bar(x + offsets[0], df_merged['Sequencial'], width, label='Sequencial')
rects_all.append(rects_seq)
for idx, p in enumerate(proc_counts):
    pos = x + offsets[idx + 1]
    vals = df_merged[p].values
    rects = ax.bar(pos, vals, width, label=f'{p} Processos')
    rects_all.append(rects)
ax.set_ylabel('Tempo de Execução (segundos) - Escala Logarítmica')
ax.set_xlabel('Tamanho da Matriz (n x n)')
ax.set_title('Comparação de Tempo de Execução (Sequencial vs. MPI)')
ax.set_xticks(x)
ax.set_xticklabels(tamanhos)
ax.legend()
ax.set_yscale('log')
for rects in rects_all:
    labels = [f'{r.get_height():.2f}' if np.isfinite(r.get_height()) else '' for r in rects]
    ax.bar_label(rects, labels=labels, padding=3)
fig.tight_layout()
plt.savefig("grafico_tempo_execucao_mpi.png")
print("Gráfico de Tempo de Execução salvo como 'grafico_tempo_execucao_mpi.png'")


# Calcula e Gerar o Gráfico de Speedup 

for p in proc_counts:
    df_merged[f'Speedup_{p}'] = df_merged['Sequencial'] / df_merged[p]
print("\nCálculo do Speedup:")
speedup_cols = [f'Speedup_{p}' for p in proc_counts]
print(df_merged[['Tamanho'] + speedup_cols])
print("------------------------------------------\n")
fig2, ax2 = plt.subplots(figsize=(12, 8))
num_procs_axis = [1] + proc_counts
for index, row in df_merged.iterrows():
    tamanho = row['Tamanho']
    speedups = [1.0]
    for p in proc_counts:
        val = row.get(f'Speedup_{p}', np.nan)
        speedups.append(val if np.isfinite(val) else np.nan)
    ax2.plot(num_procs_axis, speedups, marker='o', linestyle='-', label=f'Matriz {tamanho}x{tamanho}')
ax2.plot(num_procs_axis, num_procs_axis, linestyle='--', color='k', label='Speedup Ideal (Linear)')
ax2.set_ylabel('Speedup (Sequencial / Paralelo)')
ax2.set_xlabel('Número de Processos (MPI)')
ax2.set_title('Escalabilidade e Speedup do Código MPI')
ax2.legend()
ax2.grid(True, which="both", ls="--")
ax2.set_xticks(num_procs_axis)
plt.savefig("grafico_speedup_mpi.png")
print("Gráfico de Speedup salvo como 'grafico_speedup_mpi.png'")


# Calcula e Gerar o Gráfico de Eficiência 

print("\nCálculo da Eficiência:")
eficiencia_cols = []
# Calcula as colunas de eficiência
for p in proc_counts:
    col_name = f'Eficiencia_{p}'
    # Eficiência = Speedup / P
    df_merged[col_name] = df_merged[f'Speedup_{p}'] / p
    eficiencia_cols.append(col_name)

print(df_merged[['Tamanho'] + eficiencia_cols])
print("------------------------------------------\n")

# Prepara dados para o gráfico de linhas (Eficiência)
fig3, ax3 = plt.subplots(figsize=(12, 8))

# Plota uma linha para cada tamanho de matriz
for index, row in df_merged.iterrows():
    tamanho = row['Tamanho']
    # Eficiência com 1 processo é sempre 1.0 (Speedup=1 / P=1)
    eficiencias = [1.0] 
    for p in proc_counts:
        val = row.get(f'Eficiencia_{p}', np.nan)
        eficiencias.append(val if np.isfinite(val) else np.nan)
    ax3.plot(num_procs_axis, eficiencias, marker='o', linestyle='-', label=f'Matriz {tamanho}x{tamanho}')

# Plota a linha de eficiência ideal (1.0) para referência
ax3.axhline(y=1.0, linestyle='--', color='k', label='Eficiência Ideal (1.0)')

# Adiciona títulos e labels
ax3.set_ylabel('Eficiência (Speedup / Processos)')
ax3.set_xlabel('Número de Processos (MPI)')
ax3.set_title('Eficiência Paralela do Código MPI')
ax3.legend()
ax3.grid(True, which="both", ls="--")
ax3.set_xticks(num_procs_axis)
# Define o limite inferior do eixo Y para 0
ax3.set_ylim(bottom=0)

plt.savefig("grafico_eficiencia_mpi.png")
print("Gráfico de Eficiência salvo como 'grafico_eficiencia_mpi.png'")

# Exibe todos os gráficos
plt.show()
