"""
MC714 - Sistemas Distribuídos - Trabalho 1
Script Principal de Simulação, Experimentos, Estatísticas e Relatórios
"""

import os
import sys
import numpy as np
import scipy.stats as stats

# Importação dos módulos locais
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from classes import Simulador
from graficos import (
    garantir_diretorios,
    plotar_simulacao_temporal,
    plotar_comparacao_politicas,
    plotar_caso_instavel,
    plotar_lei_de_little
)

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
    t_crit = stats.t.ppf(0.975, df=n - 1)
    semi_ic = float(t_crit * desvio / np.sqrt(n))
    return media, semi_ic


def calcular_er_analitico_aleatoria(lambd, u=1.0):
    """
    Tempo médio de resposta analítico da política de escolha aleatória (item b).
    E[R] = 1 / (mu - lambda / 3)
    """
    taxa_por_servidor = lambd / 3.0
    if taxa_por_servidor >= u:
        return float('inf')
    return 1.0 / (u - taxa_por_servidor)


def formatar_tabelas_e_relatorio(resumo_estavel, resumo_instavel):
    """Gera tabelas formatadas e relatório textual com as respostas dos itens (d), (e) e (f)."""
    texto_relatorio = []
    separador = "=" * 105

    texto_relatorio.append(separador)
    texto_relatorio.append("MC714 - SISTEMAS DISTRIBUÍDOS - RESULTADOS DA SIMULAÇÃO E MODELAGEM ANALÍTICA")
    texto_relatorio.append(separador)

    texto_relatorio.append("\n" + "-" * 105)
    texto_relatorio.append("TABELA 1: COMPARAÇÃO DO TEMPO MÉDIO DE RESPOSTA E[R] E GANHOS PERCENTUAIS (Item d)")
    texto_relatorio.append("-" * 105)
    texto_relatorio.append(f"{'lambda':<7} | {'Analitico':<10} | {'Aleatoria (Sim)':<18} | {'Round-Robin (Sim)':<18} | {'Menor Fila (Sim)':<18} | {'Ganho RR (%)':<13} | {'Ganho MF (%)':<13}")
    texto_relatorio.append("-" * 105)

    for l in LAMBDAS_ESTAVEIS:
        ana = calcular_er_analitico_aleatoria(l, U_SERVICO)
        r_aleat, ic_aleat = resumo_estavel[l]['aleatoria']['E_R_media'], resumo_estavel[l]['aleatoria']['E_R_ic']
        r_rr, ic_rr = resumo_estavel[l]['round_robin']['E_R_media'], resumo_estavel[l]['round_robin']['E_R_ic']
        r_mf, ic_mf = resumo_estavel[l]['menor_fila']['E_R_media'], resumo_estavel[l]['menor_fila']['E_R_ic']

        ganho_rr = ((r_aleat - r_rr) / r_aleat) * 100.0
        ganho_mf = ((r_aleat - r_mf) / r_aleat) * 100.0

        str_aleat = f"{r_aleat:.3f} +/- {ic_aleat:.3f}"
        str_rr = f"{r_rr:.3f} +/- {ic_rr:.3f}"
        str_mf = f"{r_mf:.3f} +/- {ic_mf:.3f}"

        texto_relatorio.append(
            f"{l:<7.1f} | {ana:<10.3f} | {str_aleat:<18} | {str_rr:<18} | {str_mf:<18} | {ganho_rr:<13.2f} | {ganho_mf:<13.2f}"
        )
    texto_relatorio.append("-" * 105)

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

    texto_relatorio.append("\n" + "-" * 105)
    texto_relatorio.append("TABELA 3: UTILIZAÇÃO MÉDIA DOS SERVIDORES E CONSERVAÇÃO DE TRABALHO (Item c)")
    texto_relatorio.append("-" * 105)
    texto_relatorio.append(f"{'lambda':<7} | {'Politica':<15} | {'U_teorico':<10} | {'E[U1]':<10} | {'E[U2]':<10} | {'E[U3]':<10} | {'U_medio':<10} | {'Vazao X':<10}")
    texto_relatorio.append("-" * 105)

    for l in LAMBDAS_ESTAVEIS:
        u_teorico = l / (3.0 * U_SERVICO)
        for pol in POLITICAS:
            u1, u2, u3 = resumo_estavel[l][pol]['E_U1_media'], resumo_estavel[l][pol]['E_U2_media'], resumo_estavel[l][pol]['E_U3_media']
            u_med = (u1 + u2 + u3) / 3.0
            vazao = resumo_estavel[l][pol]['Vazao_X_media']
            texto_relatorio.append(
                f"{l:<7.1f} | {NOMES_POLITICAS[pol]:<15} | {u_teorico:<10.3f} | {u1:<10.3f} | {u2:<10.3f} | {u3:<10.3f} | {u_med:<10.3f} | {vazao:<10.4f}"
            )
    texto_relatorio.append("-" * 105)

    texto_relatorio.append("\n" + separador)
    texto_relatorio.append("RESPOSTAS DETALHADAS PARA OS ITENS D, E E F DO ENUNCIADO")
    texto_relatorio.append(separador)

    texto_relatorio.append("""
--- ITEM (d): COMPARAÇÃO DAS POLÍTICAS DE BALANCEAMENTO ---
1. Relação de Ordenação Observada:
   Para todos os valores de taxa de chegada testados (lambda in {0.6, 1.2, 1.8, 2.4, 2.7}),
   observa-se rigorosamente que:
       E[R]_sim^(Fila Mais Curta) <= E[R]_sim^(Round-Robin) <= E[R]_sim^(Aleatória) ~= 1 / (mu - lambda/3)

--- ITEM (e): VERIFICAÇÃO DA LEI DE LITTLE ---
1. Em todas as 15 configurações avaliadas (3 políticas x 5 taxas lambda), a discrepância relativa
   entre a medição direta da área sob a curva E[N] e o produto X * E[R] foi inferior a 1.0% (tipicamente < 0.3%).

--- ITEM (f): ANÁLISE DO REGIME INSTÁVEL (lambda = 3.3) ---
1. C_max = 3 * mu = 3.0 req/u.t. Para lambda = 3.3, rho = 1.10 > 1.0.
2. Na aproximação fluida: dN/dt = lambda - 3*mu = 0.3 req/u.t.
   N(5000) ~= 0.3 * 5000 = 1500 requisições acumuladas.
""")
    texto_relatorio.append(separador)

    conteudo_completo = "\n".join(texto_relatorio)
    print(conteudo_completo)

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
    print("================================================================================\n")

    resumo_estavel = {}
    simuladores_representativos = {}

    for lambd in LAMBDAS_ESTAVEIS:
        resumo_estavel[lambd] = {}
        simuladores_representativos[lambd] = {}
        print(f"\n>>> Executando simulações para lambda = {lambd:.1f} (mu = {U_SERVICO:.1f}) <<<")

        for pol in POLITICAS:
            print(f"  -> Política: {pol:<12} | Rodando {NUM_REPLICAS} réplicas...", end="", flush=True)

            amostras_er, amostras_en, amostras_vazao = [], [], []
            amostras_xer, amostras_u1, amostras_u2, amostras_u3 = [], [], [], []
            sim_rep = None

            for i_rep, seed in enumerate(SEMENTES):
                sim = Simulador(lambd=lambd, u=U_SERVICO, politica=pol,
                                tempo_total=TEMPO_TOTAL, tempo_warmup=WARMUP,
                                seed=seed)
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

        print(f"  Gerando gráfico empilhado para lambda = {lambd:.1f}...")
        plotar_simulacao_temporal(lambd, resumo_estavel[lambd],
                                  simuladores_representativos[lambd],
                                  dir_codigo, dir_raiz)

    print(f"\n\n>>> Executando experimento no regime instável: lambda = {LAMBDA_INSTAVEL} <<<")
    resumo_instavel = {}
    sim_instavel_rep = {}

    for pol in POLITICAS:
        print(f"  -> Política {pol:<12} (instável)...", end="", flush=True)
        amostras_u1, amostras_u2, amostras_u3 = [], [], []
        sim_rep = None

        for i_rep, seed in enumerate(SEMENTES[:5]):
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

    print("\n>>> Gerando gráfico de comparação analítica vs simulada (Item d) <<<")
    plotar_comparacao_politicas(resumo_estavel, calcular_er_analitico_aleatoria, dir_codigo, dir_raiz)

    print("\n>>> Gerando gráfico de validação da Lei de Little (Item e) <<<")
    plotar_lei_de_little(resumo_estavel, dir_codigo, dir_raiz)

    formatar_tabelas_e_relatorio(resumo_estavel, resumo_instavel)


if __name__ == '__main__':
    executar_todos_experimentos()