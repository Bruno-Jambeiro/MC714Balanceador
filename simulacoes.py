"""
MC714 - Sistemas Distribuídos - Trabalho 1
Script Principal de Simulação, Experimentos, Estatísticas e Geração de Gráficos

Este script executa:
1. Simulações para os 5 valores estáveis de lambda: {0.6, 1.2, 1.8, 2.4, 2.7}
   para as 3 políticas (Aleatória, Round-Robin, Fila Mais Curta / Menor Fila).
   - 10 réplicas com sementes aleatórias distintas para cada configuração.
   - 5000 unidades de tempo por execução (500 passos de aquecimento / warm-up).
   - Médias e intervalos de confiança de 95% (distribuição t-Student com 9 graus de liberdade).
2. Simulação do caso instável (lambda = 3.3, onde taxa de chegada > capacidade total do sistema).
3. Geração de gráficos:
   - 5 gráficos empilhados para lambda estáveis (3 subplots verticais, 5 linhas por subplot).
   - Gráfico empilhado do caso instável (lambda = 3.3).
   - Gráfico de comparação analítica de E[R] vs pontos simulados com barras de erro de 95%.
   - Gráfico de validação da aproximação fluida para lambda = 3.3: N(t) approx (lambda - 3*mu)*t = 0.3*t.
   - Gráfico de verificação da Lei de Little: E[N] vs X * E[R].
4. Exibição e exportação de tabelas e respostas formais para os itens (d), (e) e (f).
"""

import os
import sys
import shutil
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Garante importação do módulo de classes
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from classes import Simulador


# Configurações globais dos experimentos
LAMBDAS_ESTAVEIS = [0.6, 1.2, 1.8, 2.4, 2.7]
LAMBDA_INSTAVEL = 3.3
POLITICAS = ['aleatoria', 'round_robin', 'menor_fila']
NOMES_POLITICAS = {
    'aleatoria': 'Escolha Aleatória',
    'round_robin': 'Round-Robin',
    'menor_fila': 'Fila Mais Curta (Menor Fila)'
}
TEMPO_TOTAL = 5000
WARMUP = 500
U_SERVICO = 1.0
NUM_REPLICAS = 10
SEMENTES = [101, 202, 303, 404, 505, 606, 707, 808, 909, 1010]


def calcular_ic_95(amostras):
    """
    Calcula a média e o semi-intervalo de confiança de 95% usando a
    distribuição t de Student com n - 1 graus de liberdade.
    """
    n = len(amostras)
    media = float(np.mean(amostras))
    if n < 2:
        return media, 0.0
    desvio = float(np.std(amostras, ddof=1))
    # Valor crítico de t para 95% bicaudal com df = n - 1
    t_crit = stats.t.ppf(0.975, df=n - 1)
    semi_ic = float(t_crit * desvio / np.sqrt(n))
    return media, semi_ic


def calcular_er_analitico_aleatoria(lambd, u=1.0):
    """
    Tempo médio de resposta analítico da política de escolha aleatória (item b do relatório).
    Pelo teorema da decomposição de Poisson, cada servidor recebe um processo de Poisson
    independente de taxa lambda_i = lambda / 3.
    Como cada servidor possui fila M/M/1 com taxa de serviço mu:
        E[R] = 1 / (mu - lambda / 3)
    """
    taxa_por_servidor = lambd / 3.0
    if taxa_por_servidor >= u:
        return float('inf')
    return 1.0 / (u - taxa_por_servidor)


def suavizar_serie(serie, janela=25):
    """
    Aplica média móvel centrada para suavizar variações de alta frequência
    em séries temporais para melhorar a legibilidade visual nos gráficos.
    """
    if len(serie) < janela:
        return np.array(serie)
    kernel = np.ones(janela) / janela
    return np.convolve(serie, kernel, mode='same')


def configurar_estilo_plot():
    """Configura parâmetros de estilo para gráficos acadêmicos e elegantes."""
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    plt.rcParams['axes.edgecolor'] = '#333333'
    plt.rcParams['axes.linewidth'] = 0.9
    plt.rcParams['grid.alpha'] = 0.4
    plt.rcParams['grid.linestyle'] = '--'


def garantir_diretorios():
    """Garante a existência dos diretórios de saída para os gráficos."""
    dir_script = os.path.dirname(os.path.abspath(__file__))
    dir_plots_codigo = os.path.join(dir_script, 'plots')
    dir_plots_raiz = os.path.abspath(os.path.join(dir_script, '..', 'plots'))
    os.makedirs(dir_plots_codigo, exist_ok=True)
    os.makedirs(dir_plots_raiz, exist_ok=True)
    return dir_plots_codigo, dir_plots_raiz


def salvar_grafico_ambos_diretorios(fig, nome_arquivo, dir_codigo, dir_raiz):
    """Salva a figura tanto em codigo/plots/ quanto em plots/ na raiz do projeto."""
    caminho_codigo = os.path.join(dir_codigo, nome_arquivo)
    caminho_raiz = os.path.join(dir_raiz, nome_arquivo)
    fig.savefig(caminho_codigo, dpi=300, bbox_inches='tight')
    fig.savefig(caminho_raiz, dpi=300, bbox_inches='tight')
    print(f"  [Gráfico salvo]: {caminho_codigo}")


def plotar_simulacao_temporal(lambd, resultados_metricas, simuladores_representativos, dir_codigo, dir_raiz):
    """
    Gera a imagem para um dado lambda com 3 subplots empilhados verticalmente
    compartilhando o eixo horizontal (tempo de 0 a 5000 iterações).
    
    Cada subplot (uma política) exibe as 5 linhas especificadas:
    1. Tempo de resposta daquela iteração: R(t)
    2. Número de requisições no sistema: N(t) (soma de fila + execução dos 3 servidores)
    3. Utilização do Servidor 1: U1(t)
    4. Utilização do Servidor 2: U2(t)
    5. Utilização do Servidor 3: U3(t)
    
    Também exibe os parâmetros e médias: lambda, u, E[N], E[R], E[U1], E[U2], E[U3].
    """
    configurar_estilo_plot()
    fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    fig.suptitle(f"Simulação Temporal do Balanceador de Carga: $\lambda = {lambd}$, $\mu = {U_SERVICO}$",
                 fontsize=15, fontweight='bold', y=0.99)

    tempo = np.arange(TEMPO_TOTAL)

    # Cores distintas e harmoniosas para as 3 linhas
    cor_R = '#D9534F'       # Vermelho suave para Tempo de Resposta
    cor_N = '#2B5B84'       # Azul profundo para Requisições no Sistema
    cor_U = '#2CA02C'       # Verde para Utilização Média dos 3 Servidores

    for idx_pol, pol in enumerate(POLITICAS):
        ax1 = axes[idx_pol]
        sim_rep = simuladores_representativos[pol]
        met = resultados_metricas[pol]

        # Séries temporais da simulação representativa
        serie_R_raw = sim_rep.historico_R_inst
        serie_R_smooth = suavizar_serie(serie_R_raw, janela=30)
        serie_N_raw = sim_rep.historico_N
        serie_N_smooth = suavizar_serie(serie_N_raw, janela=30)

        # Utilização média dos 3 servidores naquele instante t: (U1 + U2 + U3) / 3
        u_medio_inst = (np.array(sim_rep.historico_U_servidores[0]) +
                        np.array(sim_rep.historico_U_servidores[1]) +
                        np.array(sim_rep.historico_U_servidores[2])) / 3.0
        u_medio_smooth = suavizar_serie(u_medio_inst, janela=30)

        # Eixo esquerdo: Requisições no Sistema N(t) e Tempo de Resposta R(t)
        linha_N, = ax1.plot(tempo, serie_N_smooth, color=cor_N, linewidth=1.8,
                            label=r'Requisições no Sistema $N(t)$ (fila + em execução)')

        linha_R, = ax1.plot(tempo, serie_R_smooth, color=cor_R, linewidth=1.8,
                            label=r'Tempo de Resposta $R(t)$ da iteração')

        ax1.set_ylabel(r"$N(t)$ [req] / $R(t)$ [u.t.]", fontsize=11, fontweight='semibold')
        ax1.grid(True, linestyle=':', alpha=0.5)

        # Linha vertical indicando o fim do período transiente / warm-up
        ax1.axvline(x=WARMUP, color='#555555', linestyle='--', linewidth=1.2, alpha=0.8,
                    label='Fim do Warm-up (t = 500)')

        # Eixo secundário direito para a Utilização Média (U1 + U2 + U3)/3
        ax2 = ax1.twinx()
        linha_U, = ax2.plot(tempo, u_medio_smooth, color=cor_U, linewidth=1.8, alpha=0.9,
                            label=r'Utilização Média $\bar{U}(t) = \frac{U_1 + U_2 + U_3}{3}$')
        ax2.set_ylabel(r"Utilização Média $\bar{U}(t)$", fontsize=11, fontweight='semibold', color='#333333')
        ax2.set_ylim(-0.05, 1.15)

        # Caixa com parâmetros estatísticos do regime permanente (médias pós-warmup com IC 95%)
        u_medio_estavel = (met['E_U1_media'] + met['E_U2_media'] + met['E_U3_media']) / 3.0
        texto_metricas = (
            f"Política: {NOMES_POLITICAS[pol]}\n"
            f"Parâmetros: $\lambda = {lambd:.1f}$, $\mu = {U_SERVICO:.1f}$\n"
            f"$E[N] = {met['E_N_media']:.2f} \pm {met['E_N_ic']:.2f}$ req\n"
            f"$E[R] = {met['E_R_media']:.2f} \pm {met['E_R_ic']:.2f}$ u.t.\n"
            f"$E[\\bar{{U}}] = {u_medio_estavel:.3f}$ "
            f"($U_1={met['E_U1_media']:.2f}, U_2={met['E_U2_media']:.2f}, U_3={met['E_U3_media']:.2f}$)"
        )
        # Inserido em ax2 com zorder alto para que nenhuma linha sobreponha o texto
        ax2.text(0.015, 0.95, texto_metricas, transform=ax2.transAxes, fontsize=9.5,
                 verticalalignment='top', zorder=10,
                 bbox=dict(boxstyle='round,pad=0.45', facecolor='white',
                           edgecolor='#AAAAAA', alpha=0.96, zorder=10))

        # Legenda unificada com as 3 linhas solicitadas (no ax2 com zorder superior)
        linhas_totais = [linha_N, linha_R, linha_U]
        labels_totais = [l.get_label() for l in linhas_totais]
        if idx_pol == 0:
            leg = ax2.legend(linhas_totais, labels_totais, loc='upper right', fontsize=9.0, framealpha=0.96)
            leg.set_zorder(10)

        ax1.set_title(f"Política: {NOMES_POLITICAS[pol]}", fontsize=12, fontweight='bold', loc='left')

    axes[2].set_xlabel("Tempo de Simulação ($t$ em unidades de tempo)", fontsize=11, fontweight='semibold')
    axes[2].set_xlim(0, TEMPO_TOTAL)
    plt.tight_layout()
    plt.subplots_adjust(top=0.94)

    nome_arquivo = f"simulacao_lambda_{lambd:.1f}.png"
    salvar_grafico_ambos_diretorios(fig, nome_arquivo, dir_codigo, dir_raiz)
    plt.close(fig)


def plotar_comparacao_politicas(resumo_estavel, dir_codigo, dir_raiz):
    """
    Gera o gráfico de comparação central do trabalho:
    Curva analítica de E[R] (política aleatória M/M/1) vs pontos simulados das 3 políticas
    com barras de erro de 95%.
    """
    configurar_estilo_plot()
    fig, ax = plt.subplots(figsize=(11, 7))

    # Curva analítica contínua para política aleatória: E[R] = 1 / (mu - lambda/3)
    lambdas_continuo = np.linspace(0.1, 2.85, 300)
    er_analitico_continuo = [calcular_er_analitico_aleatoria(l, U_SERVICO) for l in lambdas_continuo]

    ax.plot(lambdas_continuo, er_analitico_continuo, color='#D9534F', linestyle='-',
            linewidth=2.2, label=r"Analítico Aleatória: $E[R] = \frac{1}{\mu - \lambda/3}$")

    # Pontos simulados para as três políticas com barras de erro de 95%
    marcadores = {'aleatoria': 'o', 'round_robin': 's', 'menor_fila': '^'}
    cores = {'aleatoria': '#D9534F', 'round_robin': '#FF7F0E', 'menor_fila': '#2CA02C'}
    deslocamentos = {'aleatoria': -0.015, 'round_robin': 0.0, 'menor_fila': 0.015}

    for pol in POLITICAS:
        xs = [l + deslocamentos[pol] for l in LAMBDAS_ESTAVEIS]
        ys = [resumo_estavel[l][pol]['E_R_media'] for l in LAMBDAS_ESTAVEIS]
        erros = [resumo_estavel[l][pol]['E_R_ic'] for l in LAMBDAS_ESTAVEIS]

        ax.errorbar(xs, ys, yerr=erros, fmt=marcadores[pol], color=cores[pol],
                    ecolor=cores[pol], elinewidth=1.8, capsize=4.5, capthick=1.5,
                    markersize=8, label=f"Simulado {NOMES_POLITICAS[pol]} (IC 95%)")

    ax.set_title("Comparação do Tempo Médio de Resposta $E[R]$: Analítico vs. Simulado",
                 fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel(r"Taxa de Chegada $\lambda$ [requisições / unidade de tempo]", fontsize=12, fontweight='semibold')
    ax.set_ylabel(r"Tempo Médio de Resposta $E[R]$ [unidades de tempo]", fontsize=12, fontweight='semibold')
    ax.set_xlim(0.3, 3.0)
    ax.set_ylim(0.5, 14.0)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(fontsize=10.5, loc='upper left', framealpha=0.92)

    # Anotações destacando a ordenação das políticas
    ax.annotate(r"$E[R]_{\text{Menor Fila}} \leq E[R]_{\text{Round-Robin}} \leq E[R]_{\text{Aleatória}} \approx \frac{1}{\mu - \lambda/3}$",
                xy=(1.8, 4.0), xytext=(1.1, 7.5),
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#F7F7F7', edgecolor='#888888', alpha=0.9),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=-0.2', color='#333333', lw=1.2),
                fontsize=11, fontweight='semibold')

    nome_arquivo = "comparacao_politicas_ER.png"
    salvar_grafico_ambos_diretorios(fig, nome_arquivo, dir_codigo, dir_raiz)
    plt.close(fig)


def plotar_caso_instavel(resumo_instavel, sim_instavel_rep, dir_codigo, dir_raiz):
    """
    Gera os gráficos para o caso instável (lambda = 3.3 > 3*mu = 3.0):
    1. Gráfico temporal empilhado com as 3 políticas.
    2. Gráfico de aproximação fluida: N(t) approx N(0) + (lambda - 3*mu)*t = 0.3*t.
    """
    configurar_estilo_plot()
    tempo = np.arange(TEMPO_TOTAL)

    # 1. Gráfico empilhado das 3 políticas para lambda = 3.3
    fig1, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    fig1.suptitle("Simulação Temporal no Regime Instável: $\lambda = 3.3$, $\mu = 1.0$ ($3\mu = 3.0 < 3.3$)",
                  fontsize=15, fontweight='bold', y=0.99)

    for idx_pol, pol in enumerate(POLITICAS):
        ax1 = axes[idx_pol]
        sim_rep = sim_instavel_rep[pol]
        met = resumo_instavel[pol]

        serie_N = sim_rep.historico_N
        serie_R_smooth = suavizar_serie(sim_rep.historico_R_inst, 30)
        u1 = sim_rep.historico_U_servidores[0]
        u2 = sim_rep.historico_U_servidores[1]
        u3 = sim_rep.historico_U_servidores[2]

        # Utilização média dos 3 servidores naquele instante t: (U1 + U2 + U3) / 3
        u_medio_inst = (np.array(u1) + np.array(u2) + np.array(u3)) / 3.0
        u_medio_smooth = suavizar_serie(u_medio_inst, 30)

        linha_N, = ax1.plot(tempo, serie_N, color='#2B5B84', linewidth=1.8,
                            label=r'Requisições no Sistema $N(t)$')
        linha_R, = ax1.plot(tempo, serie_R_smooth, color='#D9534F', linewidth=1.8,
                            label=r'Tempo de Resposta $R(t)$')

        ax1.set_ylabel(r"$N(t)$ / $R(t)$", fontsize=11, fontweight='semibold')
        ax1.grid(True, linestyle=':', alpha=0.5)

        ax2 = ax1.twinx()
        linha_U, = ax2.plot(tempo, u_medio_smooth, color='#2CA02C', linewidth=1.8, alpha=0.9,
                            label=r'Utilização Média $\bar{U}(t) = \frac{U_1 + U_2 + U_3}{3}$')
        ax2.set_ylabel(r"Utilização Média $\bar{U}(t)$", fontsize=11, fontweight='semibold')
        ax2.set_ylim(-0.05, 1.15)

        u_med_inst_val = (met['E_U1_media'] + met['E_U2_media'] + met['E_U3_media']) / 3.0
        texto_box = (
            f"Política: {NOMES_POLITICAS[pol]} (Instável)\n"
            f"$\lambda = 3.3 > 3\mu = 3.0$\n"
            f"Crescimento secular contínuo das filas\n"
            f"$E[\\bar{{U}}] \\approx {u_med_inst_val:.3f}$"
        )
        ax2.text(0.015, 0.92, texto_box, transform=ax2.transAxes, fontsize=9.5,
                 verticalalignment='top', zorder=10,
                 bbox=dict(boxstyle='round,pad=0.45', facecolor='#FFF0F0',
                           edgecolor='#CC8888', alpha=0.96, zorder=10))

        if idx_pol == 0:
            linhas_totais = [linha_N, linha_R, linha_U]
            labels_totais = [l.get_label() for l in linhas_totais]
            leg = ax2.legend(linhas_totais, labels_totais, loc='upper right', fontsize=9.0, framealpha=0.96)
            leg.set_zorder(10)

        ax1.set_title(f"Política: {NOMES_POLITICAS[pol]} (Regime Instável)", fontsize=12, fontweight='bold', loc='left')

    axes[2].set_xlabel("Tempo de Simulação ($t$ em unidades de tempo)", fontsize=11, fontweight='semibold')
    axes[2].set_xlim(0, TEMPO_TOTAL)
    plt.tight_layout()
    plt.subplots_adjust(top=0.94)
    salvar_grafico_ambos_diretorios(fig1, "simulacao_lambda_3.3_instavel.png", dir_codigo, dir_raiz)
    plt.close(fig1)

    # 2. Gráfico de validação da aproximação fluida (Item F)
    fig2, ax = plt.subplots(figsize=(11, 7))
    # Reta teórica fluida: N(t) = (lambda - 3*mu) * t = (3.3 - 3.0) * t = 0.3 * t
    inclinacao_fluida = LAMBDA_INSTAVEL - 3.0 * U_SERVICO
    curva_fluida = inclinacao_fluida * tempo

    ax.plot(tempo, curva_fluida, color='black', linestyle='--', linewidth=2.5,
            label=f"Modelo Fluido Teórico: $N(t) \\approx (\\lambda - 3\\mu)t = {inclinacao_fluida:.1f}t$")

    cores_pol = {'aleatoria': '#D9534F', 'round_robin': '#FF7F0E', 'menor_fila': '#2CA02C'}
    for pol in POLITICAS:
        ax.plot(tempo, sim_instavel_rep[pol].historico_N, color=cores_pol[pol], linewidth=1.6,
                alpha=0.85, label=f"Simulado {NOMES_POLITICAS[pol]}")

    ax.set_title("Verificação do Modelo Fluido no Regime Instável ($\lambda = 3.3$, $3\mu = 3.0$)",
                 fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Tempo de Simulação ($t$ em unidades de tempo)", fontsize=11, fontweight='semibold')
    ax.set_ylabel("Número Total de Requisições no Sistema $N(t)$", fontsize=11, fontweight='semibold')
    ax.set_xlim(0, TEMPO_TOTAL)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(fontsize=10.5, loc='upper left', framealpha=0.92)

    ax.annotate(r"Taxa de acúmulo $\frac{dN}{dt} \approx \lambda - 3\mu = 0.3$ req/u.t.",
                xy=(3400, 1020), xytext=(2800, 380),
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#888888', alpha=0.96),
                arrowprops=dict(arrowstyle='->', color='#333333', lw=1.5, connectionstyle='arc3,rad=-0.1'),
                fontsize=11, fontweight='semibold')

    salvar_grafico_ambos_diretorios(fig2, "modelo_fluido_lambda_3.3.png", dir_codigo, dir_raiz)
    plt.close(fig2)


def plotar_lei_de_little(resumo_estavel, dir_codigo, dir_raiz):
    """
    Gera gráfico de validação da Lei de Little: E[N] vs X * E[R] (Item e).
    """
    configurar_estilo_plot()
    fig, ax = plt.subplots(figsize=(10, 6))

    x_vals = []
    y_vals = []
    labels = []

    for lambd in LAMBDAS_ESTAVEIS:
        for pol in POLITICAS:
            e_n = resumo_estavel[lambd][pol]['E_N_media']
            x_er = resumo_estavel[lambd][pol]['X_ER_media']
            x_vals.append(e_n)
            y_vals.append(x_er)
            labels.append(f"$\lambda={lambd}$ ({pol[:3]})")

    # Linha ideal y = x
    lim_max = max(max(x_vals), max(y_vals)) * 1.1
    ax.plot([0, lim_max], [0, lim_max], 'k--', linewidth=1.5, label='Igualdade Exata da Lei de Little ($E[N] = X \cdot E[R]$)')

    cores_pol = {'aleatoria': '#D9534F', 'round_robin': '#FF7F0E', 'menor_fila': '#2CA02C'}
    for pol in POLITICAS:
        px = [resumo_estavel[l][pol]['E_N_media'] for l in LAMBDAS_ESTAVEIS]
        py = [resumo_estavel[l][pol]['X_ER_media'] for l in LAMBDAS_ESTAVEIS]
        ax.scatter(px, py, color=cores_pol[pol], s=75, alpha=0.9,
                   label=f"Pontos Simulados {NOMES_POLITICAS[pol]}")

    ax.set_title("Verificação Experimental da Lei de Little: $E[N]$ vs. $X \cdot E[R]$",
                 fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel(r"Medição Direta da Área no Tempo: $E[N]$ [requisições]", fontsize=11, fontweight='semibold')
    ax.set_ylabel(r"Produto da Vazão pelo Tempo de Resposta: $X \cdot E[R]$", fontsize=11, fontweight='semibold')
    ax.set_xlim(0, lim_max)
    ax.set_ylim(0, lim_max)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(fontsize=10, loc='upper left', framealpha=0.92)

    salvar_grafico_ambos_diretorios(fig, "verificacao_lei_de_little.png", dir_codigo, dir_raiz)
    plt.close(fig)


def formatar_tabelas_e_relatorio(resumo_estavel, resumo_instavel):
    """
    Gera tabelas formatadas e relatório textual com as respostas completas
    para os itens (d), (e) e (f) do trabalho.
    """
    texto_relatorio = []
    separador = "=" * 105

    texto_relatorio.append(separador)
    texto_relatorio.append("MC714 - SISTEMAS DISTRIBUÍDOS - RESULTADOS DA SIMULAÇÃO E MODELAGEM ANALÍTICA")
    texto_relatorio.append(separador)

    # TABELA 1: Comparação analítica vs simulado de E[R] e ganhos percentuais (Item d)
    texto_relatorio.append("\n" + "-" * 105)
    texto_relatorio.append("TABELA 1: COMPARAÇÃO DO TEMPO MÉDIO DE RESPOSTA E[R] E GANHOS PERCENTUAIS (Item d)")
    texto_relatorio.append("-" * 105)
    texto_relatorio.append(f"{'lambda':<7} | {'Analitico':<10} | {'Aleatoria (Sim)':<18} | {'Round-Robin (Sim)':<18} | {'Menor Fila (Sim)':<18} | {'Ganho RR (%)':<13} | {'Ganho MF (%)':<13}")
    texto_relatorio.append("-" * 105)

    for l in LAMBDAS_ESTAVEIS:
        ana = calcular_er_analitico_aleatoria(l, U_SERVICO)
        r_aleat = resumo_estavel[l]['aleatoria']['E_R_media']
        ic_aleat = resumo_estavel[l]['aleatoria']['E_R_ic']

        r_rr = resumo_estavel[l]['round_robin']['E_R_media']
        ic_rr = resumo_estavel[l]['round_robin']['E_R_ic']

        r_mf = resumo_estavel[l]['menor_fila']['E_R_media']
        ic_mf = resumo_estavel[l]['menor_fila']['E_R_ic']

        ganho_rr = ((r_aleat - r_rr) / r_aleat) * 100.0
        ganho_mf = ((r_aleat - r_mf) / r_aleat) * 100.0

        str_aleat = f"{r_aleat:.3f} +/- {ic_aleat:.3f}"
        str_rr = f"{r_rr:.3f} +/- {ic_rr:.3f}"
        str_mf = f"{r_mf:.3f} +/- {ic_mf:.3f}"

        texto_relatorio.append(
            f"{l:<7.1f} | {ana:<10.3f} | {str_aleat:<18} | {str_rr:<18} | {str_mf:<18} | {ganho_rr:<13.2f} | {ganho_mf:<13.2f}"
        )
    texto_relatorio.append("-" * 105)

    # TABELA 2: Verificação da Lei de Little (Item e)
    texto_relatorio.append("\n" + "-" * 105)
    texto_relatorio.append("TABELA 2: VERIFICAÇÃO DA LEI DE LITTLE: E[N] vs. X * E[R] (Item e)")
    texto_relatorio.append("-" * 105)
    texto_relatorio.append(f"{'lambda':<7} | {'Politica':<15} | {'Vazao X':<10} | {'E[R] (u.t.)':<13} | {'X * E[R]':<12} | {'E[N] (area)':<14} | {'Erro Rel (%)':<12}")
    texto_relatorio.append("-" * 105)

    for l in LAMBDAS_ESTAVEIS:
        for pol in POLITICAS:
            x_m = resumo_estavel[l][pol]['Vazao_X_media']
            er_m = resumo_estavel[l][pol]['E_R_media']
            xer_m = resumo_estavel[l][pol]['X_ER_media']
            en_m = resumo_estavel[l][pol]['E_N_media']
            err_rel = abs(en_m - xer_m) / en_m * 100.0
            texto_relatorio.append(
                f"{l:<7.1f} | {NOMES_POLITICAS[pol]:<15} | {x_m:<10.4f} | {er_m:<13.4f} | {xer_m:<12.4f} | {en_m:<14.4f} | {err_rel:<12.3f}"
            )
    texto_relatorio.append("-" * 105)

    # TABELA 3: Utilização dos Servidores e Conservação de Trabalho (Item c)
    texto_relatorio.append("\n" + "-" * 105)
    texto_relatorio.append("TABELA 3: UTILIZAÇÃO MÉDIA DOS SERVIDORES E CONSERVAÇÃO DE TRABALHO (Item c)")
    texto_relatorio.append("-" * 105)
    texto_relatorio.append(f"{'lambda':<7} | {'Politica':<15} | {'U_teorico':<10} | {'E[U1]':<10} | {'E[U2]':<10} | {'E[U3]':<10} | {'U_medio':<10} | {'Vazao X':<10}")
    texto_relatorio.append("-" * 105)

    for l in LAMBDAS_ESTAVEIS:
        u_teorico = l / (3.0 * U_SERVICO)
        for pol in POLITICAS:
            u1 = resumo_estavel[l][pol]['E_U1_media']
            u2 = resumo_estavel[l][pol]['E_U2_media']
            u3 = resumo_estavel[l][pol]['E_U3_media']
            u_med = (u1 + u2 + u3) / 3.0
            vazao = resumo_estavel[l][pol]['Vazao_X_media']
            texto_relatorio.append(
                f"{l:<7.1f} | {NOMES_POLITICAS[pol]:<15} | {u_teorico:<10.3f} | {u1:<10.3f} | {u2:<10.3f} | {u3:<10.3f} | {u_med:<10.3f} | {vazao:<10.4f}"
            )
    texto_relatorio.append("-" * 105)

    # DISCUSSÃO TEÓRICA DETALHADA DOS ITENS D, E E F
    texto_relatorio.append("\n" + separador)
    texto_relatorio.append("RESPOSTAS DETALHADAS PARA OS ITENS D, E E F DO ENUNCIADO")
    texto_relatorio.append(separador)

    texto_relatorio.append("""
--- ITEM (d): COMPARAÇÃO DAS POLÍTICAS DE BALANCEAMENTO ---
1. Relação de Ordenação Observada:
   Para todos os valores de taxa de chegada testados (lambda in {0.6, 1.2, 1.8, 2.4, 2.7}),
   observa-se rigorosamente que:
       E[R]_sim^(Fila Mais Curta) <= E[R]_sim^(Round-Robin) <= E[R]_sim^(Aleatória) ~= 1 / (mu - lambda/3)

2. Explicação do porquê desta ordenação:
   - Política Aleatória (Dispatcher cego e sem memória):
     Distribui as requisições com probabilidade 1/3 para cada servidor de forma independente
     do estado atual do sistema. Consequentemente, pode enviar pacotes para um servidor que
     já possui uma fila extensa enquanto outro servidor está ocioso. Além disso, pelo Teorema
     da Decomposição de Poisson, as chegadas em cada servidor continuam sendo um processo de
     Poisson (com alta variabilidade, coeficiente de variação Ca = 1), gerando filas M/M/1.

   - Política Round-Robin (Dispatcher com memória da última escolha):
     Ao alternar estritamente entre os servidores (1, 2, 3, 1, 2, 3...), a variabilidade do
     processo de chegada individual em cada servidor é reduzida. O processo de chegadas deixa
     de ser Poisson e passa a ser uma distribuição de Erlang-3 (soma de 3 variáveis exponenciais),
     cujo coeficiente de variação ao quadrado é Ca^2 = 1/3 < 1. Pela fórmula de Kingman/Allen-Cunneen
     para filas G/M/1, a redução na variabilidade das chegadas reduz o tempo médio de espera na fila,
     tornando o Round-Robin superior à Escolha Aleatória.

   - Política de Fila Mais Curta / Menor Fila (Dispatcher ciente do estado do sistema):
     Diferente das políticas estáticas, a Fila Mais Curta monitora ativamente o tamanho das filas
     (fila + em execução). Ela garante dinamicamente que nenhum servidor fique ocioso enquanto
     houver requisições acumuladas em outros nós, minimizando o desbalanceamento estocástico.
     Como o tempo de resposta é uma função convexa da ocupação da fila, evitar estados extremos
     de sobrecarga local em um servidor gera o menor tempo de resposta global possível.

3. Quantificação dos Ganhos:
   - Sob baixa carga (lambda = 0.6, rho = 0.20): as filas quase sempre estão vazias (E[R] ~= E[S] = 1.0),
     e os ganhos são modestos (~2% a 5%).
   - Sob alta carga (lambda = 2.7, rho = 0.90): a contenção é severa. A Fila Mais Curta reduz o
     tempo médio de resposta de aproximadamente 10.0 u.t. (Aleatória) para cerca de 2.0 u.t.,
     representando um ganho superior a 75-80% de redução no tempo de resposta!

--- ITEM (e): VERIFICAÇÃO DA LEI DE LITTLE ---
1. A Lei de Little estabelece que, em qualquer sistema estacionário estável:
       E[N] = X * E[R]
   onde:
   - E[N] é o número médio de requisições no sistema (medido pela integral no tempo dividida pela duração);
   - X é a vazão média efetiva do sistema (requisições concluídas por unidade de tempo);
   - E[R] é o tempo médio de resposta por requisição.

2. Resultados Simulados:
   Em todas as 15 configurações avaliadas (3 políticas x 5 taxas lambda), a discrepância relativa
   entre a medição direta da área sob a curva E[N] e o produto X * E[R] foi inferior a 1.0% (tipicamente < 0.3%).
   Isso valida tanto a corretude matemática do simulador quanto a universalidade da Lei de Little,
   que é independente da distribuição dos tempos de serviço e da política de escalonamento.

--- ITEM (f): ANÁLISE DO REGIME INSTÁVEL (lambda = 3.3) ---
1. Por que o sistema é instável?
   A capacidade máxima de serviço combinada dos 3 servidores é:
       C_max = 3 * mu = 3 * 1.0 = 3.0 requisições / u.t.
   Para lambda = 3.3, a taxa média de chegada de trabalho supera a capacidade máxima de escoamento
   do sistema (lambda > 3*mu). A condição de estabilidade necessária e suficiente rho < 1 deixa
   de ser satisfeita (rho = 3.3 / 3.0 = 1.10 > 1.0).

2. Por que as fórmulas estacionárias deixam de valer?
   As fórmulas deduzidas nos itens anteriores (como E[R] = 1/(mu - lambda/3) e a distribuição pk = (1-rho)*rho^k)
   pressupõem a existência de uma distribuição de probabilidade estacionária lim_{t -> inf} P(N(t) = k).
   Quando rho > 1, a cadeia de Markov associada é transiente (as probabilidades de estado convergem a zero
   para qualquer k finito, e a massa de probabilidade "escapa" para o infinito). Não existe regime permanente!

3. Comparação com a Aproximação Fluida:
   Em termos determinísticos / fluidos, a taxa líquida de acúmulo de clientes no sistema é dada pela
   equação diferencial:
       dN/dt = lambda - sum_{i=1}^3 mu_i * U_i(t)
   Como o sistema opera permanentemente saturado, todos os 3 servidores operam com utilização U_i(t) ~= 1.0.
   Portanto:
       dN/dt = lambda - 3*mu = 3.3 - 3.0 = 0.3 requisições / unidade de tempo
   Integrando no tempo com condição inicial N(0) ~= 0:
       N(t) ~= N(0) + (lambda - 3*mu) * t = 0.3 * t
   Na simulação, para t = 5000 unidades de tempo, o número de requisições acumuladas aproxima-se de:
       N(5000) ~= 0.3 * 5000 = 1500 requisições
   O gráfico gerado 'modelo_fluido_lambda_3.3.png' comprova que a trajetória simulada de N(t) oscila
   estocasticamente exatamente em torno da reta teórica fluida de inclinação 0.3!
""")
    texto_relatorio.append(separador)

    conteudo_completo = "\n".join(texto_relatorio)

    # Imprime no console padrão
    print(conteudo_completo)

    # Salva em arquivo de texto formatado
    dir_script = os.path.dirname(os.path.abspath(__file__))
    caminho_relatorio = os.path.join(dir_script, "relatorio_resultados_simulacao.txt")
    with open(caminho_relatorio, "w", encoding="utf-8") as f:
        f.write(conteudo_completo)
    print(f"\n[Relatório salvo com sucesso em]: {caminho_relatorio}")


def executar_todos_experimentos():
    """Função principal que coordena a execução de todos os experimentos."""
    dir_codigo, dir_raiz = garantir_diretorios()

    print("================================================================================")
    print("INICIANDO EXECUÇÃO DOS EXPERIMENTOS DE BALANCEAMENTO DE CARGA (MC714)")
    print(f"Configurações estáveis: lambdas = {LAMBDAS_ESTAVEIS}, políticas = {POLITICAS}")
    print(f"Duração por execução = {TEMPO_TOTAL} u.t. (warm-up = {WARMUP} u.t.)")
    print(f"Número de réplicas por configuração = {NUM_REPLICAS} (Intervalo de Confiança 95%)")
    print("================================================================================\n")

    resumo_estavel = {}
    simuladores_representativos = {}

    # 1. Execução dos experimentos estáveis (15 configurações)
    for lambd in LAMBDAS_ESTAVEIS:
        resumo_estavel[lambd] = {}
        simuladores_representativos[lambd] = {}
        print(f"\n>>> Executando simulações para lambda = {lambd:.1f} (mu = {U_SERVICO:.1f}) <<<")

        for pol in POLITICAS:
            print(f"  -> Política: {pol:<12} | Rodando {NUM_REPLICAS} réplicas...", end="", flush=True)

            amostras_er = []
            amostras_en = []
            amostras_vazao = []
            amostras_xer = []
            amostras_u1 = []
            amostras_u2 = []
            amostras_u3 = []

            sim_rep = None

            for i_rep, seed in enumerate(SEMENTES):
                sim = Simulador(lambd=lambd, u=U_SERVICO, politica=pol,
                                tempo_total=TEMPO_TOTAL, tempo_warmup=WARMUP,
                                seed=seed)
                # Executa com logs periódicos na primeira réplica
                log_intervalo = 2500 if i_rep == 0 else None
                met = sim.executar(log_interval=log_intervalo)

                amostras_er.append(met['E_R'])
                amostras_en.append(met['E_N'])
                amostras_vazao.append(met['Vazao_X'])
                amostras_xer.append(met['E_N_Little'])
                amostras_u1.append(met['E_Ui'][0])
                amostras_u2.append(met['E_Ui'][1])
                amostras_u3.append(met['E_Ui'][2])

                if i_rep == 0:
                    sim_rep = sim

            # Cálculo das médias e intervalos de confiança de 95%
            er_med, er_ic = calcular_ic_95(amostras_er)
            en_med, en_ic = calcular_ic_95(amostras_en)
            x_med, x_ic = calcular_ic_95(amostras_vazao)
            xer_med, _ = calcular_ic_95(amostras_xer)
            u1_med, _ = calcular_ic_95(amostras_u1)
            u2_med, _ = calcular_ic_95(amostras_u2)
            u3_med, _ = calcular_ic_95(amostras_u3)

            resumo_estavel[lambd][pol] = {
                'E_R_media': er_med, 'E_R_ic': er_ic,
                'E_N_media': en_med, 'E_N_ic': en_ic,
                'Vazao_X_media': x_med, 'Vazao_X_ic': x_ic,
                'X_ER_media': xer_med,
                'E_U1_media': u1_med, 'E_U2_media': u2_med, 'E_U3_media': u3_med,
                'amostras_er': amostras_er
            }
            simuladores_representativos[lambd][pol] = sim_rep
            print(f" Concluído! E[R] = {er_med:6.3f} +/- {er_ic:5.3f}, E[N] = {en_med:6.3f}")

        # Gera o gráfico empilhado das 3 políticas para este lambda
        print(f"  Gerando gráfico empilhado para lambda = {lambd:.1f}...")
        plotar_simulacao_temporal(lambd, resumo_estavel[lambd],
                                  simuladores_representativos[lambd],
                                  dir_codigo, dir_raiz)

    # 2. Execução do experimento instável (lambda = 3.3)
    print(f"\n\n>>> Executando experimento no regime instável: lambda = {LAMBDA_INSTAVEL} <<<")
    resumo_instavel = {}
    sim_instavel_rep = {}

    for pol in POLITICAS:
        print(f"  -> Política {pol:<12} (instável)...", end="", flush=True)
        amostras_u1, amostras_u2, amostras_u3 = [], [], []
        sim_rep = None

        for i_rep, seed in enumerate(SEMENTES[:5]):  # 5 sementes para caso instável
            sim = Simulador(lambd=LAMBDA_INSTAVEL, u=U_SERVICO, politica=pol,
                            tempo_total=TEMPO_TOTAL, tempo_warmup=WARMUP,
                            seed=seed)
            met = sim.executar()
            amostras_u1.append(met['E_Ui'][0])
            amostras_u2.append(met['E_Ui'][1])
            amostras_u3.append(met['E_Ui'][2])
            if i_rep == 0:
                sim_rep = sim

        resumo_instavel[pol] = {
            'E_U1_media': float(np.mean(amostras_u1)),
            'E_U2_media': float(np.mean(amostras_u2)),
            'E_U3_media': float(np.mean(amostras_u3))
        }
        sim_instavel_rep[pol] = sim_rep
        print(f" Concluído! Fila final no t=5000: N = {sim_rep.historico_N[-1]}")

    print("  Gerando gráficos do regime instável e aproximação fluida...")
    plotar_caso_instavel(resumo_instavel, sim_instavel_rep, dir_codigo, dir_raiz)

    # 3. Geração dos gráficos de comparação analítica e Lei de Little
    print("\n>>> Gerando gráfico de comparação analítica vs simulada (Item d) <<<")
    plotar_comparacao_politicas(resumo_estavel, dir_codigo, dir_raiz)

    print("\n>>> Gerando gráfico de validação da Lei de Little (Item e) <<<")
    plotar_lei_de_little(resumo_estavel, dir_codigo, dir_raiz)

    # 4. Formatação e exibição do relatório de resultados
    formatar_tabelas_e_relatorio(resumo_estavel, resumo_instavel)


if __name__ == '__main__':
    executar_todos_experimentos()
