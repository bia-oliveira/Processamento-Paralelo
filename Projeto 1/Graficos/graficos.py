import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

try:
    # Carrega os resultados sequenciais e paralelos
    df_seq = pd.read_csv('resultados.csv')
    df_par = pd.read_csv('resultados2.csv')
except FileNotFoundError as e:
    print(f"Erro: Arquivo não encontrado. Certifique-se que o arquivo '{e.filename}' está na mesma pasta que o script.")
    exit()

# Média para a execução sequencial (agrupando por tamanho)
seq_avg = df_seq.groupby('Tamanho')['Tempo_s'].mean().reset_index()
seq_avg = seq_avg.rename(columns={'Tempo_s': 'Sequencial'})

# Média para a execução paralela (agrupando por tamanho E número de threads)
par_avg = df_par.groupby(['Tamanho', 'Threads'])['Tempo_s'].mean().reset_index()

print("Médias de tempo de execução (em segundos):")
print("------------------------------------------")
print("Versão Sequencial:")
print(seq_avg)
print("\nVersão Paralela:")
print(par_avg)
print("------------------------------------------\n")


# Preparar dados para o Gráfico de Barras 
# Pivota a tabela de médias paralelas para que cada contagem de threads seja uma coluna
par_pivot = par_avg.pivot(index='Tamanho', columns='Threads', values='Tempo_s')
# Ordena as colunas de threads (para garantir ordem 2,4,6,8,...)
threads = sorted(par_pivot.columns.tolist())
# Junta os dados sequenciais e paralelos em uma única tabela
df_merged = pd.merge(seq_avg, par_pivot, on='Tamanho')
print(df_merged)
print("------------------------------------------\n")

# --- Gera o Gráfico de Barras (Tempo de Execução) ---

tamanhos = df_merged['Tamanho'].values
x = np.arange(len(tamanhos))  # posições das labels no eixo x

# número total de barras por grupo (1 seq + N threads)
total_bars = 1 + len(threads)
width = 0.8 / total_bars  # largura adaptativa

fig, ax = plt.subplots(figsize=(14, 8))

# Lista para guardar os retângulos (para rotular)
rects_all = []

# Plota a barra para a versão sequencial (primeira barra de cada grupo)
offsets = [(i - (total_bars - 1) / 2) * width for i in range(total_bars)]
rects_seq = ax.bar(x + offsets[0], df_merged['Sequencial'], width, label='Sequencial')
rects_all.append(rects_seq)

# Plota as barras para cada configuração de threads dinamicamente
for idx, th in enumerate(threads):
    pos = x + offsets[idx + 1]
    vals = df_merged[th].values
    rects = ax.bar(pos, vals, width, label=f'{th} Threads')
    rects_all.append(rects)

# Adiciona títulos e labels
ax.set_ylabel('Tempo de Execução (segundos) - Escala Logarítmica')
ax.set_xlabel('Tamanho da Matriz (n x n)')
ax.set_title('Comparação de Tempo de Execução (Sequencial vs. Paralelo)')
ax.set_xticks(x)
ax.set_xticklabels(tamanhos)
ax.legend()

# Usa escala logarítmica para melhor visualização
ax.set_yscale('log')

# Adiciona os valores no topo das barras para clareza (formata apenas valores finitos)
for rects in rects_all:
    # cria lista de labels somente para valores finitos
    labels = []
    for r in rects:
        h = r.get_height()
        if np.isfinite(h):
            labels.append(f'{h:.2f}')
        else:
            labels.append('')
    ax.bar_label(rects, labels=labels, padding=3)

fig.tight_layout()
plt.savefig("grafico_tempo_execucao.png") # Salva o gráfico em um arquivo
print("Gráfico de Tempo de Execução salvo como 'grafico_tempo_execucao.png'")


# --- Calcula e Gerar o Gráfico de Speedup ---

# Calcula as colunas de speedup para todas as threads encontradas
for t in threads:
    df_merged[f'Speedup_{t}'] = df_merged['Sequencial'] / df_merged[t]

print("\nCálculo do Speedup:")
speedup_cols = [f'Speedup_{t}' for t in threads]
print(df_merged[['Tamanho'] + speedup_cols])
print("------------------------------------------\n")

# Prepara dados para o gráfico de linhas (Speedup)
fig2, ax2 = plt.subplots(figsize=(12, 8))

num_threads_axis = [1] + threads  # Eixo X: 1 (seq), 2, 4, 6, 8, 10, 12...

# Plota uma linha para cada tamanho de matriz
for index, row in df_merged.iterrows():
    tamanho = row['Tamanho']
    speedups = [1.0]  # Speedup com 1 thread é sempre 1
    for t in threads:
        val = row.get(f'Speedup_{t}', np.nan)
        speedups.append(val if np.isfinite(val) else np.nan)
    ax2.plot(num_threads_axis, speedups, marker='o', linestyle='-', label=f'Matriz {tamanho}x{tamanho}')

# Plota a linha de speedup ideal (linear) para referência
ax2.plot(num_threads_axis, num_threads_axis, linestyle='--', color='k', label='Speedup Ideal (Linear)')

# Adiciona títulos e labels
ax2.set_ylabel('Speedup (Sequencial / Paralelo)')
ax2.set_xlabel('Número de Threads')
ax2.set_title('Escalabilidade e Speedup do Código Paralelo')
ax2.legend()
ax2.grid(True, which="both", ls="--")
ax2.set_xticks(num_threads_axis) # Garante que todos os pontos de thread apareçam no eixo x

plt.savefig("grafico_speedup.png") # Salva o gráfico em um arquivo
print("Gráfico de Speedup salvo como 'grafico_speedup.png'")

# Exibe os gráficos
plt.show()
# ...existing code...
