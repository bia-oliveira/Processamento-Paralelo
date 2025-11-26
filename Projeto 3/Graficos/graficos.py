import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

files = ['resultados_cuda1.csv', 'resultados_cuda2.csv', 'resultados_cuda3.csv']
dfs = []

print("--- Carregando arquivos ---")
try:
    for f in files:
        # Tenta ler o arquivo
        df_temp = pd.read_csv(f)
        dfs.append(df_temp)
        print(f"Leitura de '{f}': OK ({len(df_temp)} linhas)")

    # Junta todos os dataframes em um só grande dataframe
    df_all = pd.concat(dfs)

except FileNotFoundError as e:
    print(f"\nERRO CRÍTICO: Arquivo '{e.filename}' não encontrado.")
    print("Certifique-se de que os 3 arquivos .csv estão na mesma pasta do script.")
    exit()

# Agrupa por Tamanho e Metodo e tira a média aritmética das 3 rodadas
df_avg = df_all.groupby(['Tamanho', 'Metodo'])[['Tempo_s', 'GFLOPS']].mean().reset_index()

print("\n--- Médias Consolidadas ---")
print(df_avg)
print("-------------------------------------\n")

# Pivotar tabelas para facilitar os cálculos de Speedup e plotagem
df_pivot_time = df_avg.pivot(index='Tamanho', columns='Metodo', values='Tempo_s')
df_pivot_gflops = df_avg.pivot(index='Tamanho', columns='Metodo', values='GFLOPS')

# Configuração visual dos gráficos
plt.style.use('ggplot')
sns.set_theme(style="whitegrid")
colors = {'Sequencial': '#1f77b4', 'CUDA_Naive': '#ff7f0e', 'CUDA_Tiled': '#2ca02c'}
markers = {'Sequencial': 'X', 'CUDA_Naive': 'o', 'CUDA_Tiled': 's'}

# GRÁFICO 1: Tempo de Execução (Logarítmico)
plt.figure(figsize=(10, 6))

# Barplot para comparar tempos absolutos
ax = sns.barplot(data=df_avg, x='Tamanho', y='Tempo_s', hue='Metodo', palette=colors)

# Escala Logarítmica
plt.yscale('log') 
plt.ylabel('Tempo de Execução (s) - Escala Log')
plt.xlabel('Tamanho da Matriz (N)')
plt.title('Comparação de Tempo: CPU Sequencial vs GPU CUDA')
plt.legend(title='Implementação')

plt.tight_layout()
plt.savefig("grafico_tempo_cuda.png", dpi=300)
print("-> Gerado: grafico_tempo_cuda.png")
plt.close() # Fecha a figura para liberar memória e evitar sobreposição

# GRÁFICO 2: Speedup (Relativo ao Sequencial)
# Calcular Speedup: T_seq / T_metodo
df_pivot_time['Speedup_Naive'] = df_pivot_time['Sequencial'] / df_pivot_time['CUDA_Naive']
df_pivot_time['Speedup_Tiled'] = df_pivot_time['Sequencial'] / df_pivot_time['CUDA_Tiled']

print("\n--- Tabela de Speedup (Médio) ---")
print(df_pivot_time[['Speedup_Naive', 'Speedup_Tiled']])
print("---------------------------------\n")

plt.figure(figsize=(10, 6))

# Plotar linhas de Speedup
plt.plot(df_pivot_time.index, df_pivot_time['Speedup_Naive'], 
         marker=markers['CUDA_Naive'], linewidth=2.5, label='GPU Naive', color=colors['CUDA_Naive'])
plt.plot(df_pivot_time.index, df_pivot_time['Speedup_Tiled'], 
         marker=markers['CUDA_Tiled'], linewidth=2.5, label='GPU Tiled (Shared Mem)', color=colors['CUDA_Tiled'])

# Linha de referência (Speedup = 1.0)
plt.axhline(y=1.0, color='black', linestyle='--', label='Baseline (Sequencial)', alpha=0.7)

plt.title('Speedup: Ganho de Desempenho em relação à CPU')
plt.ylabel('Speedup (x vezes mais rápido)')
plt.xlabel('Tamanho da Matriz (N)')
plt.legend()
plt.grid(True, which='both', linestyle='--', alpha=0.7)
plt.xticks(df_pivot_time.index)

plt.tight_layout()
plt.savefig("grafico_speedup_cuda.png", dpi=300)
print("-> Gerado: grafico_speedup_cuda.png")
plt.close()

# GRÁFICO 3: Escalabilidade de Desempenho (GFLOPS vs Tamanho)
plt.figure(figsize=(10, 6))

# Plotar GFLOPS para cada método
for metodo in ['Sequencial', 'CUDA_Naive', 'CUDA_Tiled']:
    # Garante que existe dados para o método antes de plotar
    if metodo in df_pivot_gflops.columns:
        plt.plot(df_pivot_gflops.index, df_pivot_gflops[metodo], 
                 marker=markers[metodo], linewidth=3, label=metodo, color=colors[metodo], markersize=8)

plt.title('Escalabilidade de Desempenho (GFLOPS x Tamanho da Matriz)')
plt.ylabel('Desempenho (GFLOPS)')
plt.xlabel('Tamanho da Matriz (N)')
plt.legend(title='Implementação', loc='upper left')
plt.grid(True, which='both', linestyle='--', alpha=0.7)
plt.xticks(df_pivot_gflops.index)

# Anotação explicativa sobre o pico da GPU
if 'CUDA_Tiled' in df_pivot_gflops.columns:
    max_gflops = float(df_pivot_gflops['CUDA_Tiled'].max()) # Conversão explícita para float
    # Pega o maior índice (tamanho de matriz) que tenha esse valor máximo
    # Assumindo que o max ocorre em 4096, mas vamos garantir pegando o índice correto
    max_idx = df_pivot_gflops['CUDA_Tiled'].idxmax()

    plt.annotate(f'Pico GPU: {max_gflops:.1f} GFLOPS', 
                 xy=(max_idx, max_gflops), 
                 xytext=(max_idx/2, max_gflops), # Posiciona texto um pouco à esquerda
                 arrowprops=dict(facecolor='black', shrink=0.05))

plt.tight_layout()
plt.savefig("grafico_escalabilidade_cuda.png", dpi=300)
print("-> Gerado: grafico_escalabilidade_cuda.png")
plt.close()

# GRÁFICO 4: Eficiência Computacional (Barras de GFLOPS)
plt.figure(figsize=(10, 6))

# Barplot dos GFLOPS
ax = sns.barplot(data=df_avg, x='Tamanho', y='GFLOPS', hue='Metodo', palette=colors)

plt.title('Eficiência Computacional (GFLOPS)')
plt.ylabel('GFLOPS (Bilhões de Op/s)')
plt.xlabel('Tamanho da Matriz (N)')
plt.legend(title='Implementação')

# Adicionar valores no topo das barras 
for container in ax.containers:
    ax.bar_label(container, fmt='%.1f', padding=3, fontsize=9)

plt.tight_layout()
plt.savefig("grafico_eficiencia_cuda.png", dpi=300)
print("-> Gerado: grafico_eficiencia_cuda.png")
plt.close()

print("\nProcesso finalizado com sucesso!")
