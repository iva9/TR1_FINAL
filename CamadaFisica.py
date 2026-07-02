"""Camada Física do simulador de TR1.

Este arquivo concentra as funções de modulação, demodulação e ruído.
A interface e o socket chamam estas funções, mantendo a camada física separada
como solicitado no enunciado.
"""

import numpy as np

# Quantidade de amostras usadas para desenhar cada bit no tempo.
AMOSTRAS_POR_BIT = 100
# Frequência normalizada usada como base para clock e portadoras.
FREQUENCIA_BASE = 1


def nrz_polar(bits: str) -> list[int]:
    """Codifica bits em NRZ-Polar: 0 -> -1 e 1 -> +1."""
    return [1 if bit == "1" else -1 for bit in bits]


def manchester(bits: str, clock) -> np.ndarray:
    """Codifica Manchester combinando NRZ-Polar com clock usando XOR lógico."""
    sinal_nrz_expandido = np.repeat(nrz_polar(bits), AMOSTRAS_POR_BIT)
    sinal_manchester = np.logical_xor(sinal_nrz_expandido > 0, clock > 0.5).astype(int)
    return sinal_manchester * 2 - 1


def bipolar(bits: str) -> list[int]:
    """Codifica Bipolar: 0 -> 0 e bits 1 alternam entre +1 e -1."""
    sinal = []
    ultimo_um = None
    for bit in bits:
        if bit == "0":
            sinal.append(0)
            continue
        if ultimo_um is None or ultimo_um == -1:
            ultimo_um = 1
        else:
            ultimo_um = -1
        sinal.append(ultimo_um)
    return sinal


def gerar_tempo_clock_portadora(bits: str):
    """Gera eixo de tempo normalizado, clock e portadora base para a simulação."""
    tempo = np.linspace(0, len(bits), max(1, len(bits) * AMOSTRAS_POR_BIT))
    clock = np.tile(
        np.concatenate([np.zeros(AMOSTRAS_POR_BIT // 2), np.ones(AMOSTRAS_POR_BIT // 2)]),
        len(bits),
    )
    portadora = np.sin(2 * np.pi * FREQUENCIA_BASE * tempo)
    return tempo, clock, portadora


def ask_modulation(bits: str, portadora) -> np.ndarray:
    """Implementa ASK: bit 1 transmite portadora e bit 0 zera o sinal."""
    sinal_expandido = np.repeat(nrz_polar(bits), AMOSTRAS_POR_BIT)
    return np.where(sinal_expandido == 1, portadora, 0)


def fsk_modulation(bits: str, tempo) -> np.ndarray:
    """Implementa FSK: bit 0 usa frequência baixa e bit 1 usa frequência alta."""
    sinal_expandido = np.repeat(nrz_polar(bits), AMOSTRAS_POR_BIT)
    onda_baixa = np.sin(2 * np.pi * FREQUENCIA_BASE * tempo)
    onda_alta = np.sin(2 * np.pi * (FREQUENCIA_BASE * 2) * tempo)
    return np.where(sinal_expandido == 1, onda_alta, onda_baixa)


def banda_base_qpsk(simbolos_modulados: list[complex]) -> tuple[np.ndarray, np.ndarray]:
    """Gera a forma de onda complexa dos símbolos QPSK."""
    duracao_simbolo = 2
    quantidade_simbolos = len(simbolos_modulados)
    tempo_simbolo = np.linspace(0, duracao_simbolo, AMOSTRAS_POR_BIT)
    tempo_total = np.linspace(0, duracao_simbolo * quantidade_simbolos, quantidade_simbolos * AMOSTRAS_POR_BIT)
    forma_onda = np.zeros(len(tempo_total), dtype=complex)
    for indice, simbolo in enumerate(simbolos_modulados):
        inicio = indice * AMOSTRAS_POR_BIT
        fim = (indice + 1) * AMOSTRAS_POR_BIT
        forma_onda[inicio:fim] = simbolo * np.exp(1j * 2 * np.pi * FREQUENCIA_BASE * tempo_simbolo)
    return tempo_total, forma_onda


def qpsk_modulation(bits: str) -> tuple[np.ndarray, np.ndarray]:
    """Implementa QPSK agrupando os bits de 2 em 2."""
    bits_inteiros = [int(bit) for bit in bits]
    while len(bits_inteiros) % 2 != 0:
        bits_inteiros.append(0)
    constelacao = {
        (0, 0): complex(1, 1),
        (0, 1): complex(-1, 1),
        (1, 1): complex(-1, -1),
        (1, 0): complex(1, -1),
    }
    simbolos = []
    for indice in range(0, len(bits_inteiros), 2):
        grupo = (bits_inteiros[indice], bits_inteiros[indice + 1])
        simbolos.append(constelacao[grupo])
    return banda_base_qpsk(simbolos)


def banda_base_16qam(simbolos_modulados: list[complex]) -> tuple[np.ndarray, np.ndarray]:
    """Gera a forma de onda complexa dos símbolos 16-QAM."""
    duracao_simbolo = 1
    quantidade_simbolos = len(simbolos_modulados)
    tempo_simbolo = np.linspace(0, duracao_simbolo, AMOSTRAS_POR_BIT)
    tempo_total = np.linspace(0, duracao_simbolo * quantidade_simbolos, quantidade_simbolos * AMOSTRAS_POR_BIT)
    forma_onda = np.zeros(len(tempo_total), dtype=complex)
    for indice, simbolo in enumerate(simbolos_modulados):
        inicio = indice * AMOSTRAS_POR_BIT
        fim = (indice + 1) * AMOSTRAS_POR_BIT
        forma_onda[inicio:fim] = simbolo * np.exp(1j * 2 * np.pi * FREQUENCIA_BASE * tempo_simbolo)
    return tempo_total, forma_onda


def modulacao_16qam(bits: str) -> tuple[np.ndarray, np.ndarray]:
    """Implementa 16-QAM agrupando os bits de 4 em 4."""
    bits_inteiros = [int(bit) for bit in bits]
    while len(bits_inteiros) % 4 != 0:
        bits_inteiros.append(0)
    niveis = {0: -3, 1: -1, 2: 1, 3: 3}
    simbolos = []
    for indice in range(0, len(bits_inteiros), 4):
        b0, b1, b2, b3 = bits_inteiros[indice:indice + 4]
        indice_i = (b0 << 1) | b1
        indice_q = (b2 << 1) | b3
        simbolos.append(complex(niveis[indice_i], niveis[indice_q]))
    return banda_base_16qam(simbolos)


def adicionar_ruido_gaussiano(sinal, sigma: float):
    """Soma ruído gaussiano ao sinal modulado antes da demodulação."""
    if sigma <= 0:
        return sinal
    if np.iscomplexobj(sinal):
        return sinal + np.random.normal(0, sigma, len(sinal)) + 1j * np.random.normal(0, sigma, len(sinal))
    return sinal + np.random.normal(0, sigma, len(sinal))


def modular(bits: str, tipo_modulacao: str, tecnica: str):
    """Escolhe e executa a modulação física selecionada na interface."""
    tempo, clock, portadora = gerar_tempo_clock_portadora(bits)
    if tipo_modulacao == "Digital":
        if tecnica == "NRZ-Polar":
            return tempo, np.repeat(nrz_polar(bits), AMOSTRAS_POR_BIT)
        if tecnica == "Manchester":
            return tempo, manchester(bits, clock)
        if tecnica == "Bipolar":
            return tempo, np.repeat(bipolar(bits), AMOSTRAS_POR_BIT)
    else:
        if tecnica == "ASK":
            return tempo, ask_modulation(bits, portadora)
        if tecnica == "FSK":
            return tempo, fsk_modulation(bits, tempo)
        if tecnica == "QPSK":
            return qpsk_modulation(bits)
        if tecnica == "16-QAM":
            return modulacao_16qam(bits)
    raise ValueError("Técnica de modulação desconhecida.")


def demodular_nrz_polar(sinal_modulado, quantidade_bits: int) -> str:
    """Demodula NRZ-Polar dividindo o sinal em intervalos de bit e decidindo 1 para média
    positiva e 0 para média negativa.
    """
    bits = []
    for indice in range(quantidade_bits):
        trecho = sinal_modulado[indice * AMOSTRAS_POR_BIT:(indice + 1) * AMOSTRAS_POR_BIT]
        bits.append("1" if np.mean(trecho) > 0 else "0")
    return "".join(bits)


def demodular_manchester(sinal_modulado, clock, quantidade_bits: int) -> str:
    """Demodula Manchester aplicando XOR lógico com o clock para recuperar o NRZ original e
    depois decide cada bit pela média do intervalo.
    """
    sinal_logico = sinal_modulado > 0
    clock_logico = clock > 0.5
    nrz_recuperado = np.logical_xor(sinal_logico, clock_logico)
    bits = []
    for indice in range(quantidade_bits):
        trecho = nrz_recuperado[indice * AMOSTRAS_POR_BIT:(indice + 1) * AMOSTRAS_POR_BIT]
        bits.append("1" if np.mean(trecho) >= 0.5 else "0")
    return "".join(bits)


def demodular_bipolar(sinal_modulado, quantidade_bits: int) -> str:
    """Demodula Bipolar usando o módulo da média de cada intervalo, pois o bit 1 pode
    aparecer como +1 ou -1.
    """
    bits = []
    for indice in range(quantidade_bits):
        trecho = sinal_modulado[indice * AMOSTRAS_POR_BIT:(indice + 1) * AMOSTRAS_POR_BIT]
        bits.append("1" if abs(np.mean(trecho)) > 0.5 else "0")
    return "".join(bits)


def demodular_ask(sinal_modulado, quantidade_bits: int) -> str:
    """Demodula ASK calculando a energia de cada intervalo; energia alta indica portadora
    presente e, portanto, bit 1.
    """
    energias = []
    for indice in range(quantidade_bits):
        trecho = sinal_modulado[indice * AMOSTRAS_POR_BIT:(indice + 1) * AMOSTRAS_POR_BIT]
        energias.append(float(np.mean(np.array(trecho) ** 2)))
    limite = (max(energias) + min(energias)) / 2 if energias else 0
    return "".join("1" if energia > limite else "0" for energia in energias)


def demodular_fsk(sinal_modulado, tempo, quantidade_bits: int) -> str:
    """Demodula FSK comparando cada trecho do sinal com referências de frequência baixa e
    alta por correlação.
    """
    bits = []
    for indice in range(quantidade_bits):
        inicio = indice * AMOSTRAS_POR_BIT
        fim = (indice + 1) * AMOSTRAS_POR_BIT
        trecho_sinal = sinal_modulado[inicio:fim]
        trecho_tempo = tempo[inicio:fim]
        referencia_baixa = np.sin(2 * np.pi * FREQUENCIA_BASE * trecho_tempo)
        referencia_alta = np.sin(2 * np.pi * (FREQUENCIA_BASE * 2) * trecho_tempo)
        correlacao_baixa = abs(np.sum(trecho_sinal * referencia_baixa))
        correlacao_alta = abs(np.sum(trecho_sinal * referencia_alta))
        bits.append("1" if correlacao_alta > correlacao_baixa else "0")
    return "".join(bits)


def demodular_qpsk(sinal_modulado, quantidade_bits_original: int) -> str:
    """Demodula QPSK estimando o símbolo complexo recebido e escolhendo o ponto mais
    próximo da constelação.
    """
    constelacao_inversa = {
        complex(1, 1): "00",
        complex(-1, 1): "01",
        complex(-1, -1): "11",
        complex(1, -1): "10",
    }
    duracao_simbolo = 2
    quantidade_simbolos = int(np.ceil(quantidade_bits_original / 2))
    tempo_simbolo = np.linspace(0, duracao_simbolo, AMOSTRAS_POR_BIT)
    portadora_conjugada = np.exp(-1j * 2 * np.pi * FREQUENCIA_BASE * tempo_simbolo)
    bits = ""
    for indice in range(quantidade_simbolos):
        inicio = indice * AMOSTRAS_POR_BIT
        fim = (indice + 1) * AMOSTRAS_POR_BIT
        trecho = sinal_modulado[inicio:fim]
        simbolo_estimado = np.mean(trecho * portadora_conjugada)
        ponto_mais_proximo = min(constelacao_inversa.keys(), key=lambda ponto: abs(simbolo_estimado - ponto))
        bits += constelacao_inversa[ponto_mais_proximo]
    return bits[:quantidade_bits_original]


def demodular_16qam(sinal_modulado, quantidade_bits_original: int) -> str:
    """Demodula 16-QAM estimando o símbolo complexo recebido e aproximando I e Q para os
    níveis permitidos mais próximos.
    """
    niveis_para_bits = {-3: "00", -1: "01", 1: "10", 3: "11"}
    niveis = list(niveis_para_bits.keys())
    duracao_simbolo = 1
    quantidade_simbolos = int(np.ceil(quantidade_bits_original / 4))
    tempo_simbolo = np.linspace(0, duracao_simbolo, AMOSTRAS_POR_BIT)
    portadora_conjugada = np.exp(-1j * 2 * np.pi * FREQUENCIA_BASE * tempo_simbolo)
    bits = ""
    for indice in range(quantidade_simbolos):
        inicio = indice * AMOSTRAS_POR_BIT
        fim = (indice + 1) * AMOSTRAS_POR_BIT
        trecho = sinal_modulado[inicio:fim]
        simbolo_estimado = np.mean(trecho * portadora_conjugada)
        nivel_i = min(niveis, key=lambda nivel: abs(simbolo_estimado.real - nivel))
        nivel_q = min(niveis, key=lambda nivel: abs(simbolo_estimado.imag - nivel))
        bits += niveis_para_bits[nivel_i] + niveis_para_bits[nivel_q]
    return bits[:quantidade_bits_original]


def demodular(sinal_recebido, tempo, tipo_modulacao: str, tecnica: str, quantidade_bits: int) -> str:
    """Escolhe e executa a demodulação física correspondente."""
    if tipo_modulacao == "Digital":
        _, clock, _ = gerar_tempo_clock_portadora("0" * quantidade_bits)
        if tecnica == "NRZ-Polar":
            return demodular_nrz_polar(sinal_recebido, quantidade_bits)
        if tecnica == "Manchester":
            return demodular_manchester(sinal_recebido, clock, quantidade_bits)
        if tecnica == "Bipolar":
            return demodular_bipolar(sinal_recebido, quantidade_bits)
    else:
        if tecnica == "ASK":
            return demodular_ask(sinal_recebido, quantidade_bits)
        if tecnica == "FSK":
            return demodular_fsk(sinal_recebido, tempo, quantidade_bits)
        if tecnica == "QPSK":
            return demodular_qpsk(sinal_recebido, quantidade_bits)
        if tecnica == "16-QAM":
            return demodular_16qam(sinal_recebido, quantidade_bits)
    raise ValueError("Técnica de demodulação desconhecida.")
