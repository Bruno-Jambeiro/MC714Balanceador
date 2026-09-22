"""
MC714 - Sistemas Distribuídos - Trabalho 1
Módulo de Geração de Gráficos e Visualização de Dados
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# Configurações padrão de plotagem
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
    """
    configurar_estilo_plot()
    fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    fig.suptitle(f"Simulação Temporal do Balanceador de Carga: $\lambda = {lambd}$, $\mu = {U_SERVICO}$",
                 fontsize=15, fontweight='bold', y=0.99)

    tempo = np.arange(TEMPO_TOTAL)

    cor_R = '#D9534F'       # Vermelho suave para Tempo de Resposta
    cor_N = '#2B5B84'       # Azul profundo para Requisições no Sistema
    cor_U = '#2CA02C'       # Verde para Utilização Média dos 3 Servidores

    for idx_pol, pol in enumerate(POLITICAS):
        ax1 = axes[idx_pol]
        sim_rep = simuladores_representativos[pol]
        met = resultados_metricas[pol]

        serie_R_raw = sim_rep.historico_R_inst
        serie_R_smooth = suavizar_serie(serie_R_raw, janela=30)
        serie_N_raw = sim_rep.historico_N
        serie_N_smooth = suavizar_serie(serie_N_raw, janela=30)

        u_medio_inst = (np.array(sim_rep.historico_U_servidores[0]) +
                        np.array(sim_rep.historico_U_servidores[1]) +
                        np.array(sim_rep.historico_U_servidores[2])) / 3.0
        u_medio_smooth = suavizar_serie(u_medio_inst, janela=30)

        linha_N, = ax1.plot(tempo, serie_N_smooth, color=cor_N, linewidth=1.8,
                            label=r'Requisições no Sistema $N(t)$ (fila + em execução)')

        linha_R, = ax1.plot(tempo, serie_R_smooth, color=cor_R, linewidth=1.8,
                            label=r'Tempo de Resposta $R(t)$ da iteração')

        ax1.set_ylabel(r"$N(t)$ [req] / $R(t)$ [u.t.]", fontsize=11, fontweight='semibold')
        ax1.grid(True, linestyle=':', alpha=0.5)

        ax1.axvline(x=WARMUP, color='#555555', linestyle='--', linewidth=1.2, alpha=0.8,
                    label='Fim do Warm-up (t = 500)')

        ax2 = ax1.twinx()
        linha_U, = ax2.plot(tempo, u_medio_smooth, color=cor_U, linewidth=1.8, alpha=0.9,
                            label=r'Utilização Média $\bar{U}(t) = \frac{U_1 + U_2 + U_3}{3}$')
        ax2.set_ylabel(r"Utilização Média $\bar{U}(t)$", fontsize=11, fontweight='semibold', color='#333333')
        ax2.set_ylim(-0.05, 1.15)

        u_medio_estavel = (met['E_U1_media'] + met['E_U2_media'] + met['E_U3_media']) / 3.0
        texto_metricas = (
            f"Política: {NOMES_POLITICAS[pol]}\n"
            f"Parâmetros: $\lambda = {lambd:.1f}$, $\mu = {U_SERVICO:.1f}$\n"
            f"$E[N] = {met['E_N_media']:.2f} \pm {met['E_N_ic']:.2f}$ req\n"
            f"$E[R] = {met['E_R_media']:.2f} \pm {met['E_R_ic']:.2f}$ u.t.\n"
            f"$E[\\bar{{U}}] = {u_medio_estavel:.3f}$ "
            f"($U_1={met['E_U1_media']:.2f}, U_2={met['E_U2_media']:.2f}, U_3={met['E_U3_media']:.2f}$)"
        )
        ax2.text(0.015, 0.95, texto_metricas, transform=ax2.transAxes, fontsize=9.5,
                 verticalalignment='top', zorder=10,
                 bbox=dict(boxstyle='round,pad=0.45', facecolor='white',
                           edgecolor='#AAAAAA', alpha=0.96, zorder=10))

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


def plotar_comparacao_politicas(resumo_estavel, fn_er_analitico, dir_codigo, dir_raiz):
    """
    Gera o gráfico de comparação central:
    Curva analítica de E[R] vs pontos simulados das 3 políticas com barras de erro de 95%.
    """
    configurar_estilo_plot()
    fig, ax = plt.subplots(figsize=(11, 7))

    lambdas_continuo = np.linspace(0.1, 2.85, 300)
    er_analitico_continuo = [fn_er_analitico(l, U_SERVICO) for l in lambdas_continuo]

    ax.plot(lambdas_continuo, er_analitico_continuo, color='#D9534F', linestyle='-',
            linewidth=2.2, label=r"Analítico Aleatória: $E[R] = \frac{1}{\mu - \lambda/3}$")

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

    ax.annotate(r"$E[R]_{\text{Menor Fila}} \leq E[R]_{\text{Round-Robin}} \leq E[R]_{\text{Aleatória}} \approx \frac{1}{\mu - \lambda/3}$",
                xy=(1.8, 4.0), xytext=(1.1, 7.5),
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#F7F7F7', edgecolor='#888888', alpha=0.9),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=-0.2', color='#333333', lw=1.2),
                fontsize=11, fontweight='semibold')

    nome_arquivo = "comparacao_politicas_ER.png"
    salvar_grafico_ambos_diretorios(fig, nome_arquivo, dir_codigo, dir_raiz)
    plt.close(fig)


def plotar_caso_instavel(resumo_instavel, sim_instavel_rep, dir_codigo, dir_raiz):
    """Gera os gráficos para o caso instável (lambda = 3.3 > 3*mu = 3.0)."""
    configurar_estilo_plot()
    tempo = np.arange(TEMPO_TOTAL)

    fig1, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    fig1.suptitle("Simulação Temporal no Regime Instável: $\lambda = 3.3$, $\mu = 1.0$ ($3\mu = 3.0 < 3.3$)",
                  fontsize=15, fontweight='bold', y=0.99)

    for idx_pol, pol in enumerate(POLITICAS):
        ax1 = axes[idx_pol]
        sim_rep = sim_instavel_rep[pol]
        met = resumo_instavel[pol]

        serie_N = sim_rep.historico_N
        serie_R_smooth = suavizar_serie(sim_rep.historico_R_inst, 30)
        u1, u2, u3 = sim_rep.historico_U_servidores

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

    fig2, ax = plt.subplots(figsize=(11, 7))
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
    """Gera gráfico de validação da Lei de Little: E[N] vs X * E[R]."""
    configurar_estilo_plot()
    fig, ax = plt.subplots(figsize=(10, 6))

    x_vals = []
    y_vals = []

    for lambd in LAMBDAS_ESTAVEIS:
        for pol in POLITICAS:
            e_n = resumo_estavel[lambd][pol]['E_N_media']
            x_er = resumo_estavel[lambd][pol]['X_ER_media']
            x_vals.append(e_n)
            y_vals.append(x_er)

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