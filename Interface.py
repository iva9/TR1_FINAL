"""Interface GUI em Streamlit para o simulador de TR1.

Esta interface apenas organiza a tela, chama o Simulador.py para socket/thread
e chama CamadaFisica.py e CamadaEnlace.py para executar os algoritmos.
"""

import time

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from CamadaFisica import (
    AMOSTRAS_POR_BIT,
    FREQUENCIA_BASE,
    adicionar_ruido_gaussiano,
    demodular,
    gerar_tempo_clock_portadora,
    modular,
)
from CamadaEnlace import aplicar_enlace_tx, validar_entrada
from Simulador import HOST, PORTA, RuntimeRX, enviar_objeto, iniciar_receptor


@st.cache_resource
def obter_runtime_rx() -> RuntimeRX:
    """Cria ou recupera o estado persistente do receptor RX para evitar reiniciar a thread
    a cada recarregamento do Streamlit.
    """
    return RuntimeRX()


def drenar_resultados_rx(runtime: RuntimeRX) -> None:
    """Move os resultados produzidos pela thread do receptor para o session_state do
    Streamlit para exibição na interface.
    """
    if "resultados_rx" not in st.session_state:
        st.session_state.resultados_rx = []
    while not runtime.fila_resultados.empty():
        st.session_state.resultados_rx.insert(0, runtime.fila_resultados.get())


def _expandir_bits_para_grafico(bits: str):
    """Gera eixo temporal e valores 0/1 para visualizar uma sequência binária."""
    tempo_bits = np.linspace(0, len(bits), max(1, len(bits) * AMOSTRAS_POR_BIT))
    valores_bits = np.repeat([int(bit) for bit in bits], AMOSTRAS_POR_BIT) if bits else np.array([])
    return tempo_bits, valores_bits


def plotar_visualizacao_fisica(bits_fisica: str, tempo, sinal_limpo, sinal_recebido,
                               tipo_modulacao: str, tecnica: str, bits_demodulados: str):
    """Mostra a camada física em quatro etapas: entrada, clock, canal e saída RX.

    Grandezas usadas nos gráficos:
    - eixo x: tempo normalizado t [u.t.];
    - eixo y da entrada/saída: bit b(t), adimensional, 0 ou 1;
    - eixo y do clock: amplitude normalizada do clock c(t), 0 ou 1;
    - eixo y do sinal: tensão/amplitude elétrica V(t) em volts normalizados [V].
    """
    tempo_bits, valores_bits = _expandir_bits_para_grafico(bits_fisica)
    _, clock, _ = gerar_tempo_clock_portadora(bits_fisica)
    tempo_rx, valores_rx = _expandir_bits_para_grafico(bits_demodulados)

    # Figura criada explicitamente para evitar que o Streamlit/Matplotlib reutilize
    # figuras antigas e cause gráficos espremidos ou com grande área em branco.
    fig, eixos = plt.subplots(4, 1, figsize=(9.5, 6.2), constrained_layout=False)
    fig.suptitle(
        f"Visualização da Camada Física - {tecnica} | relação: eixo x = tempo, eixo y = grandeza do sinal",
        fontsize=10,
        y=0.995,
    )

    eixos[0].step(tempo_bits, valores_bits, where="post", label="Entrada TX b(t)")
    eixos[0].set_title("Entrada binária do transmissor")
    eixos[0].set_ylabel("Bit b(t) [0 ou 1]")
    eixos[0].set_xlabel("Tempo normalizado t [u.t.]")
    eixos[0].set_ylim(-0.2, 1.2)
    eixos[0].grid(True, alpha=0.3)
    eixos[0].legend(loc="upper right")

    eixos[1].step(tempo_bits, clock, where="post", label="Clock c(t)")
    eixos[1].set_title("Clock de referência")
    eixos[1].set_ylabel("Amplitude c(t) [0 ou 1]")
    eixos[1].set_xlabel("Tempo normalizado t [u.t.]")
    eixos[1].set_ylim(-0.2, 1.2)
    eixos[1].grid(True, alpha=0.3)
    eixos[1].legend(loc="upper right")

    if np.iscomplexobj(sinal_limpo) or np.iscomplexobj(sinal_recebido):
        eixos[2].plot(tempo, np.real(sinal_limpo), label="TX limpo - I(t)", linewidth=1)
        eixos[2].plot(tempo, np.imag(sinal_limpo), label="TX limpo - Q(t)", linewidth=1)
        eixos[2].plot(tempo, np.real(sinal_recebido), label="RX com ruído - I(t)", linewidth=1, alpha=0.75)
        eixos[2].plot(tempo, np.imag(sinal_recebido), label="RX com ruído - Q(t)", linewidth=1, alpha=0.75)
        eixos[2].set_ylabel("Tensão I/Q [V]")
    else:
        eixos[2].plot(tempo, sinal_limpo, label="Sinal limpo TX V(t)", linewidth=1)
        eixos[2].plot(tempo, sinal_recebido, label="Sinal recebido RX V(t)+n(0,σ)", linewidth=1, alpha=0.75)
        eixos[2].set_ylabel("Tensão V(t) [V]")
    eixos[2].set_title("Meio de comunicação: sinal limpo x sinal recebido com ruído")
    eixos[2].set_xlabel("Tempo normalizado t [u.t.]")
    eixos[2].grid(True, alpha=0.3)
    eixos[2].legend(loc="upper right")

    eixos[3].step(tempo_rx, valores_rx, where="post", label="Bits recuperados no RX")
    eixos[3].set_title("Bits recuperados após demodulação")
    eixos[3].set_ylabel("Bit recuperado [0 ou 1]")
    eixos[3].set_xlabel("Tempo normalizado t [u.t.]")
    eixos[3].set_ylim(-0.2, 1.2)
    eixos[3].grid(True, alpha=0.3)
    eixos[3].legend(loc="upper right")

    # Reduz o tamanho dos textos para o gráfico não ocupar a tela inteira.
    for eixo in eixos:
        eixo.title.set_fontsize(9)
        eixo.xaxis.label.set_fontsize(8)
        eixo.yaxis.label.set_fontsize(8)
        eixo.tick_params(axis="both", labelsize=8)
        eixo.legend(loc="upper right", fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.955], h_pad=0.8)
    return fig


def main() -> None:
    """Monta a interface Streamlit, organiza TX/RX, recebe entradas do usuário, executa
    transmissão e mostra os resultados.
    """
    st.set_page_config(page_title="Simulador TX/RX TR1", layout="wide")
    runtime = obter_runtime_rx()
    drenar_resultados_rx(runtime)

    st.title("Simulador")
    coluna_tx, coluna_rx = st.columns([1, 1])

    with coluna_tx:
        st.header("Transmissor TX")
        tipo_entrada = st.radio("Tipo de entrada", ["Texto", "Bits"], horizontal=True)
        entrada = st.text_input("Entrada", value="menu" if tipo_entrada == "Texto" else "10110010")

        st.subheader("Camada de Enlace")
        metodo_enlace = st.selectbox(
            "Método de enquadramento/detecção/correção",
            [
                "Nenhum",
                "Contagem de Caracteres",
                "Inserção de Bytes/Caracteres com Flags",
                "Inserção de Bits com Flags",
                "Paridade Par",
                "Checksum",
                "CRC-32",
                "Hamming",
            ],
            index=6,
        )

        st.subheader("Camada Física")
        tipo_modulacao = st.radio("Tipo de modulação", ["Digital", "Portadora"], horizontal=True)
        if tipo_modulacao == "Digital":
            tecnica = st.selectbox("Técnica digital", ["NRZ-Polar", "Manchester", "Bipolar"])
        else:
            tecnica = st.selectbox("Técnica por portadora", ["ASK", "FSK", "QPSK", "16-QAM"], index=1)

        st.subheader("Meio de comunicação")
        st.write(f"**Socket TCP local:** `{HOST}:{PORTA}`")
        sigma = st.slider("Intensidade do ruído gaussiano no canal σ [V]", 0.0, 3.0, 0.0, 0.1)
        transmitir = st.button("Transmitir pelo socket", type="primary")

        if transmitir:
            if not runtime.receptor_iniciado:
                st.warning("Inicie o receptor RX antes de transmitir.")
            else:
                try:
                    bits_dados = validar_entrada(tipo_entrada, entrada)
                    bits_fisica, info_enlace = aplicar_enlace_tx(bits_dados, metodo_enlace)
                    tempo, sinal_limpo = modular(bits_fisica, tipo_modulacao, tecnica)
                    sinal_recebido = adicionar_ruido_gaussiano(sinal_limpo, sigma)
                    bits_demodulados_previa = demodular(
                        sinal_recebido,
                        tempo,
                        tipo_modulacao,
                        tecnica,
                        len(bits_fisica),
                    )

                    pacote = {
                        "tipo_entrada": tipo_entrada,
                        "tipo_modulacao": tipo_modulacao,
                        "tecnica": tecnica,
                        "metodo_enlace": metodo_enlace,
                        "quantidade_bits_fisica": len(bits_fisica),
                        "quantidade_bits_dados_original": len(bits_dados),
                        "sigma": sigma,
                        "tempo": tempo,
                        "sinal_recebido": sinal_recebido,
                        "bits_demodulados_previa": bits_demodulados_previa,
                    }
                    enviar_objeto(pacote)

                    st.session_state.ultima_tx = {
                        "bits_dados": bits_dados,
                        "bits_fisica": bits_fisica,
                        "info_enlace": info_enlace,
                        "tipo_modulacao": tipo_modulacao,
                        "tecnica": tecnica,
                        "metodo_enlace": metodo_enlace,
                        "sigma": sigma,
                        "tempo": tempo,
                        "sinal_limpo": sinal_limpo,
                        "sinal_recebido": sinal_recebido,
                        "bits_demodulados_previa": bits_demodulados_previa,
                    }
                    st.success("Pacote transmitido pelo socket para o RX.")
                    time.sleep(0.3)
                    st.rerun()
                except Exception as erro:
                    st.error(f"Erro na transmissão: {erro}")

        if "ultima_tx" in st.session_state:
            tx = st.session_state.ultima_tx
            st.subheader("Última transmissão TX")
            st.write(f"**Bits originais de dados:** `{tx['bits_dados']}`")
            st.write(f"**Método de enlace:** `{tx['metodo_enlace']}`")
            st.write(f"**Bits enviados para a física:** `{tx['bits_fisica']}`")
            st.write(f"**Modulação:** `{tx['tipo_modulacao']} / {tx['tecnica']}`")
            st.write(f"**Frequência base:** `{FREQUENCIA_BASE}`")
            if tx["tecnica"] == "FSK":
                st.write(f"**FSK baixa/alta:** `{FREQUENCIA_BASE}` e `{FREQUENCIA_BASE * 2}`")
            st.write(f"**Ruído sigma:** `{tx['sigma']}` V")
            st.info("A visualização completa da camada física aparece abaixo, em largura total da página.")

    with coluna_rx:
        st.header("Receptor RX")
        if st.button("Iniciar receptor RX", type="primary"):
            iniciar_receptor(runtime)
            time.sleep(0.2)
            st.rerun()
        st.write(f"**Endereço:** `{HOST}:{PORTA}`")
        st.write(f"**Status:** {runtime.status}")
        if runtime.erro_receptor:
            st.error(runtime.erro_receptor)
        elif runtime.receptor_iniciado:
            st.success("Thread do receptor iniciada.")
        else:
            st.warning("Clique em 'Iniciar receptor RX' antes de transmitir.")

        st.subheader("Últimas recepções RX")
        if not st.session_state.get("resultados_rx"):
            st.write("Nenhuma transmissão recebida ainda.")
        else:
            for resultado in st.session_state.resultados_rx[:5]:
                with st.expander(f"Recepção às {resultado.get('horario', '--:--:--')}", expanded=True):
                    if "erro" in resultado:
                        st.error(resultado["erro"])
                        continue
                    st.write(f"**Origem:** `{resultado['endereco']}`")
                    st.write(f"**Modulação:** `{resultado['tipo_modulacao']} / {resultado['tecnica']}`")
                    st.write(f"**Método de enlace:** `{resultado['metodo_enlace']}`")
                    st.write(f"**Ruído sigma:** `{resultado['sigma']}` V")
                    st.write(f"**Bits demodulados:** `{resultado['bits_demodulados']}`")
                    st.write(f"**Bits de dados recuperados:** `{resultado['bits_dados_recuperados']}`")
                    if resultado.get("checksum_recebido"):
                        st.write(f"**Checksum recebido:** `{resultado['checksum_recebido']}`")
                    if resultado.get("crc_recebido"):
                        st.write(f"**CRC-32 recebido:** `{resultado['crc_recebido']}`")
                    if "sindrome" in resultado:
                        st.write(f"**Síndrome Hamming:** `{resultado['sindrome']}`")
                        st.write(f"**Código corrigido:** `{resultado['codigo_corrigido']}`")
                    if resultado["valido"]:
                        st.success(resultado["mensagem_status"])
                    else:
                        st.error(resultado["mensagem_status"])
                    if resultado.get("texto_recuperado"):
                        st.write(f"**Texto recuperado:** `{resultado['texto_recuperado']}`")

    st.divider()

    if "ultima_tx" in st.session_state:
        tx = st.session_state.ultima_tx
        st.subheader("Visualização completa da Camada Física")

        fig = plotar_visualizacao_fisica(
            tx["bits_fisica"],
            tx["tempo"],
            tx["sinal_limpo"],
            tx["sinal_recebido"],
            tx["tipo_modulacao"],
            tx["tecnica"],
            tx.get("bits_demodulados_previa", ""),
        )
        st.pyplot(fig, use_container_width=False, clear_figure=True)
        plt.close(fig)


if __name__ == "__main__":
    main()
