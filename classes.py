"""
MC714 - Sistemas Distribuídos - Trabalho 1
Módulo de Classes e Estruturas de Dados do Simulador de Balanceamento de Carga

Este módulo define:
- Funções de geração estocástica (Poisson para chegadas e Exponencial para tempos de serviço).
- Classe Pacote (Requisição com timestamps e métricas).
- Classe Servidor (Fila FCFS ilimitada, 1 thread de processamento, cálculo de utilização e área sob a curva N(t)).
- Classe Balanceador (Dispatcher com políticas: Aleatória, Round-Robin e Fila Mais Curta / Menor Fila).
- Classe Simulador (Motor de simulação discreta temporal unitária, coleta de métricas e regime permanente).
"""

import numpy as np


def numero_de_pacotes_poisson(taxa_chegada):
    """
    Gera o número inteiro de requisições chegando em uma unidade de tempo através
    de uma distribuição de Poisson com taxa lambda (taxa_chegada).
    """
    numero = np.random.poisson(taxa_chegada)
    return int(numero)


def tempo_de_processamento_pacotes(n_pacotes, u):
    """
    Devolve uma lista de tamanho n_pacotes com os tempos de processamento de cada pacote.
    Os tempos são gerados por processo aleatório exponencial de parâmetro u (média E[S] = 1/u).
    """
    if n_pacotes <= 0:
        return []
    tempos = np.random.exponential(scale=1.0 / u, size=n_pacotes)
    return list(tempos)


class Pacote:
    """
    Representa uma requisição individual no sistema distribuído.
    """
    def __init__(self, id_pacote, tempo_chegada, tempo_servico):
        self.id = id_pacote
        self.tempo_chegada = float(tempo_chegada)       # Instante t em que a requisição chegou ao dispatcher
        self.tempo_servico = float(tempo_servico)       # Demanda de serviço S ~ Exp(u)
        self.tempo_restante = float(tempo_servico)      # Quantidade de serviço pendente
        self.tempo_inicio = None                        # Instante em que o servidor iniciou o atendimento
        self.tempo_conclusao = None                     # Instante em que o atendimento foi finalizado
        self.tempo_resposta = None                      # Tempo de resposta R = tempo_conclusao - tempo_chegada
        self.tempo_espera = None                        # Tempo de espera na fila W = R - S
        self.servidor_alocado = None                    # ID do servidor que atendeu a requisição

    def __repr__(self):
        return (f"Pacote(id={self.id}, chegada={self.tempo_chegada:.2f}, "
                f"servico={self.tempo_servico:.2f}, rest={self.tempo_restante:.2f})")


class Servidor:
    """
    Representa um nó servidor homogêneo com:
    - Uma única thread de processamento (atende uma requisição por vez);
    - Uma fila FCFS (First Come, First Served) ilimitada;
    - Taxa de atendimento mu = 1.0 requisição / unidade de tempo.
    """
    def __init__(self, id_servidor):
        self.id = id_servidor
        self.fila = []                                  # Fila de espera FCFS
        self.pacote_atual = None                        # Requisição em execução no núcleo
        self.total_concluidos = 0                       # Contador de requisições concluídas
        self.tempo_ocupado_acumulado = 0.0              # Tempo total em que esteve processando
        self.pacotes_finalizados = []                   # Histórico de pacotes finalizados por este servidor

    def total_requisicoes(self):
        """
        Retorna o número total de requisições no servidor:
        requisições na fila de espera + requisição atualmente em execução (0 ou 1).
        """
        return len(self.fila) + (1 if self.pacote_atual is not None else 0)

    def ocupacao_fila(self):
        """Retorna o número de requisições apenas aguardando na fila de espera."""
        return len(self.fila)

    def esta_ocupado(self):
        """Retorna True se houver requisição sendo processada no momento."""
        return self.pacote_atual is not None

    def adicionar_pacote(self, pacote):
        """Enfileira uma nova requisição na fila FCFS do servidor."""
        pacote.servidor_alocado = self.id
        self.fila.append(pacote)

    def processar(self, delta_t=1.0, tempo_atual=0.0):
        """
        Executa o trabalho do servidor no intervalo [tempo_atual, tempo_atual + delta_t].
        Consome até delta_t de processamento do pacote em execução e, caso termine,
        atende o próximo da fila imediatamente até esgotar o budget de tempo do passo.

        Calcula com exatidão a área sob a curva de ocupação N_i(t) ao longo do passo,
        conforme orientações teóricas para cálculo rigoroso de E[N] e validação da Lei de Little.

        Retorna:
            pacotes_concluidos_passo: lista de pacotes concluídos nesta iteração
            utilizacao_passo: fração de tempo do passo em que o servidor esteve ocupado (em [0.0, 1.0])
            area_ocupacao_passo: integral de N_i(t) dt ao longo do passo
        """
        budget = float(delta_t)
        tempo_ocupado_passo = 0.0
        area_ocupacao_passo = 0.0
        pacotes_concluidos_passo = []

        # Se não há pacote em execução mas há fila, promove o primeiro
        if self.pacote_atual is None and len(self.fila) > 0:
            self.pacote_atual = self.fila.pop(0)
            if self.pacote_atual.tempo_inicio is None:
                self.pacote_atual.tempo_inicio = tempo_atual

        while budget > 1e-9:
            # Número de requisições no servidor neste trecho
            n_atual = self.total_requisicoes()

            if self.pacote_atual is None:
                # Servidor totalmente ocioso pelo restante do budget
                break

            # Processa o pacote atual
            if self.pacote_atual.tempo_restante <= budget:
                # O pacote conclui dentro deste passo de tempo
                dt = self.pacote_atual.tempo_restante
                area_ocupacao_passo += n_atual * dt
                tempo_ocupado_passo += dt
                budget -= dt

                self.pacote_atual.tempo_restante = 0.0
                self.pacote_atual.tempo_conclusao = tempo_atual + tempo_ocupado_passo
                self.pacote_atual.tempo_resposta = self.pacote_atual.tempo_conclusao - self.pacote_atual.tempo_chegada
                self.pacote_atual.tempo_espera = max(0.0, self.pacote_atual.tempo_resposta - self.pacote_atual.tempo_servico)

                pacotes_concluidos_passo.append(self.pacote_atual)
                self.pacotes_finalizados.append(self.pacote_atual)
                self.total_concluidos += 1
                self.pacote_atual = None

                # Se houver outro na fila, inicia imediatamente
                if len(self.fila) > 0:
                    self.pacote_atual = self.fila.pop(0)
                    if self.pacote_atual.tempo_inicio is None:
                        self.pacote_atual.tempo_inicio = tempo_atual + tempo_ocupado_passo
            else:
                # Pacote não conclui, consome todo o restante do budget
                dt = budget
                area_ocupacao_passo += n_atual * dt
                tempo_ocupado_passo += dt
                self.pacote_atual.tempo_restante -= dt
                budget = 0.0

        self.tempo_ocupado_acumulado += tempo_ocupado_passo
        utilizacao_passo = min(1.0, tempo_ocupado_passo / delta_t)
        return pacotes_concluidos_passo, utilizacao_passo, area_ocupacao_passo

    def reset(self):
        """Reinicia as variáveis internas do servidor para uma nova execução."""
        self.fila.clear()
        self.pacote_atual = None
        self.total_concluidos = 0
        self.tempo_ocupado_acumulado = 0.0
        self.pacotes_finalizados.clear()


class Balanceador:
    """
    Balanceador de carga (dispatcher) com processamento instantâneo.
    Suporta alternância dinâmica entre três políticas:
    - 'aleatoria': Escolha equiprovável (probabilidade 1/3 para cada servidor).
    - 'round_robin': Distribuição circular (1, 2, 3, 1, 2, 3...).
    - 'menor_fila' (ou 'fila_mais_curta'): Encaminha para o servidor com menor número
      total de requisições (fila + em execução); empates são resolvidos aleatoriamente.
    """
    def __init__(self, servidores, politica='aleatoria'):
        self.servidores = servidores
        self.politica = politica
        self.ultimo_indice_rr = -1
        self.contadores_despacho = [0 for _ in range(len(servidores))]

    def definir_politica(self, nova_politica):
        """Permite alternar a política de balanceamento."""
        politicas_validas = ['aleatoria', 'round_robin', 'menor_fila', 'fila_mais_curta']
        if nova_politica not in politicas_validas:
            raise ValueError(f"Política inválida: {nova_politica}. Opções: {politicas_validas}")
        self.politica = nova_politica

    def despachar(self, pacote):
        """
        Roteia o pacote para um servidor de acordo com a política configurada.
        Retorna o índice do servidor selecionado (0, 1 ou 2).
        """
        n_servidores = len(self.servidores)

        if self.politica == 'aleatoria':
            # Escolha uniforme independente entre os servidores
            idx = int(np.random.randint(0, n_servidores))

        elif self.politica == 'round_robin':
            # Distribuição circular estrita
            idx = (self.ultimo_indice_rr + 1) % n_servidores
            self.ultimo_indice_rr = idx

        elif self.politica in ['menor_fila', 'fila_mais_curta']:
            # Consulta o número total de requisições em cada servidor (fila + em execução)
            cargas = [s.total_requisicoes() for s in self.servidores]
            min_carga = min(cargas)
            candidatos = [i for i, c in enumerate(cargas) if c == min_carga]
            if len(candidatos) == 1:
                idx = candidatos[0]
            else:
                # Empate resolvido de forma equiprovável e independente
                idx = int(np.random.choice(candidatos))
        else:
            raise ValueError(f"Política desconhecida: {self.politica}")

        self.servidores[idx].adicionar_pacote(pacote)
        self.contadores_despacho[idx] += 1
        return idx

    def reset(self):
        """Reinicia estado do balanceador."""
        self.ultimo_indice_rr = -1
        self.contadores_despacho = [0 for _ in range(len(self.servidores))]


class Simulador:
    """
    Motor de simulação para o sistema de balanceamento de carga.
    Executa a simulação por 5000 iterações (unidades de tempo), coleta
    as métricas temporais e calcula as médias em regime permanente
    (descartando os 500 primeiros passos de warm-up).
    """
    def __init__(self, lambd, u=1.0, politica='aleatoria', tempo_total=5000,
                 tempo_warmup=500, n_servidores=3, seed=None):
        self.lambd = float(lambd)
        self.u = float(u)
        self.politica = politica
        self.tempo_total = int(tempo_total)
        self.tempo_warmup = int(tempo_warmup)
        self.n_servidores = int(n_servidores)
        self.seed = seed

        if seed is not None:
            np.random.seed(seed)

        self.servidores = [Servidor(i) for i in range(self.n_servidores)]
        self.balanceador = Balanceador(self.servidores, politica=politica)

        # Séries temporais ao longo dos passos t in [0, tempo_total)
        self.historico_tempo = []
        self.historico_N = []                           # N(t): Total de requisições no sistema
        self.historico_N_servidores = [[] for _ in range(self.n_servidores)] # Ni(t)
        self.historico_filas_servidores = [[] for _ in range(self.n_servidores)] # apenas filas
        self.historico_R_inst = []                      # R(t): tempo de resposta médio dos finalizados no passo
        self.historico_U_servidores = [[] for _ in range(self.n_servidores)] # Ui(t)
        self.historico_chegadas = []                    # Chegadas em cada passo t
        self.historico_areas_passo = []                 # Área sob curva N(t) em cada passo t
        self.todos_pacotes_concluidos = []

    def executar(self, log_interval=None):
        """
        Executa o loop de simulação temporal unitária (t = 0 até tempo_total - 1).
        
        Parâmetros:
            log_interval: se fornecido (ex: 1000), imprime o estado do sistema a cada N passos.
        """
        id_global_pacote = 0
        ultimo_R = 1.0 / self.u  # Valor inicial de fallback para R(t)

        for t in range(self.tempo_total):
            self.historico_tempo.append(t)

            # 1. Chegada de pacotes no instante t segundo processo de Poisson(lambda)
            k_chegadas = numero_de_pacotes_poisson(self.lambd)
            self.historico_chegadas.append(k_chegadas)
            tempos_servico = tempo_de_processamento_pacotes(k_chegadas, self.u)

            # 2. Despacho instantâneo dos pacotes recém-chegados pelo balanceador
            for s in tempos_servico:
                p = Pacote(id_global_pacote, tempo_chegada=t, tempo_servico=s)
                id_global_pacote += 1
                self.balanceador.despachar(p)

            # 3. Processamento de 1 unidade de tempo contínua em cada um dos servidores
            concluidos_passo = []
            area_passo_total = 0.0

            for i, servidor in enumerate(self.servidores):
                concluidos_srv, u_passo, area_srv = servidor.processar(delta_t=1.0, tempo_atual=t)
                concluidos_passo.extend(concluidos_srv)
                area_passo_total += area_srv
                self.historico_U_servidores[i].append(u_passo)
                # Ocupação total do servidor no final do passo
                self.historico_N_servidores[i].append(servidor.total_requisicoes())
                self.historico_filas_servidores[i].append(servidor.ocupacao_fila())

            self.todos_pacotes_concluidos.extend(concluidos_passo)
            self.historico_areas_passo.append(area_passo_total)

            # 4. Cálculo das métricas da iteração t
            # N(t): ocupação média no passo (integral no passo / delta_t=1.0)
            n_sistema = area_passo_total
            self.historico_N.append(n_sistema)

            # R(t): tempo de resposta médio das requisições concluídas nesta iteração
            if len(concluidos_passo) > 0:
                media_r_passo = float(np.mean([p.tempo_resposta for p in concluidos_passo]))
                ultimo_R = media_r_passo
                self.historico_R_inst.append(media_r_passo)
            else:
                # Mantém o último valor observado para garantir série temporal contínua
                self.historico_R_inst.append(ultimo_R)

            # Log periódico caso configurado
            if log_interval and (t % log_interval == 0 or t == self.tempo_total - 1):
                cargas = [s.total_requisicoes() for s in self.servidores]
                filas = [s.ocupacao_fila() for s in self.servidores]
                desp = self.balanceador.contadores_despacho
                print(f"[t={t:4d}] Pol={self.politica:<11} | N_sistema={n_sistema:5.2f} | "
                      f"Filas={filas} | Total/Srv={cargas} | Despachos={desp}")

        return self.calcular_metricas_estacionarias()

    def calcular_metricas_estacionarias(self):
        """
        Calcula as métricas de desempenho em regime permanente descartando o período
        de aquecimento (warm-up das primeiras 500 unidades de tempo).
        """
        warmup = self.tempo_warmup
        duracao_estavel = self.tempo_total - warmup

        # E[N]: média temporal do número de requisições no sistema
        # Integral no tempo dividida pela duração (área sob a curva / duracao_estavel)
        total_area_pos_warmup = sum(self.historico_areas_passo[warmup:])
        e_n = float(total_area_pos_warmup / duracao_estavel)

        # E[Ui]: média temporal de utilização por servidor pós-warmup
        e_ui = [float(np.mean(self.historico_U_servidores[i][warmup:]))
                for i in range(self.n_servidores)]

        # Requisições que chegaram e concluíram após o término do período de aquecimento
        pacotes_pos_warmup = [p for p in self.todos_pacotes_concluidos
                              if p.tempo_chegada >= warmup and p.tempo_conclusao is not None]

        # E[R]: tempo médio de resposta
        if len(pacotes_pos_warmup) > 0:
            e_r = float(np.mean([p.tempo_resposta for p in pacotes_pos_warmup]))
            e_w = float(np.mean([p.tempo_espera for p in pacotes_pos_warmup]))
        else:
            e_r = 0.0
            e_w = 0.0

        # Vazão efetiva X: total de chegadas processadas pós-warmup / duracao_estavel
        vazao_x = len(pacotes_pos_warmup) / float(duracao_estavel)

        # Verificação da Lei de Little: E[N] = X * E[R]
        e_n_little = vazao_x * e_r
        erro_little = abs(e_n - e_n_little) / e_n if e_n > 0 else 0.0

        return {
            'lambda': self.lambd,
            'u': self.u,
            'politica': self.politica,
            'tempo_total': self.tempo_total,
            'warmup': self.tempo_warmup,
            'E_N': e_n,
            'E_R': e_r,
            'E_W': e_w,
            'E_Ui': e_ui,
            'Vazao_X': vazao_x,
            'E_N_Little': e_n_little,
            'Erro_Little_Rel': erro_little,
            'Despachos': list(self.balanceador.contadores_despacho),
            'Total_Concluidos': len(self.todos_pacotes_concluidos)
        }
