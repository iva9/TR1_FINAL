"""Simulador principal: socket, thread e rotina TX/RX.

Este arquivo não implementa as camadas diretamente. Ele chama:
- CamadaFisica.py para modular, adicionar ruído e demodular;
- CamadaEnlace.py para enquadrar, detectar/corrigir e desenquadrar.
"""

import pickle
import queue
import socket
import struct
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from CamadaFisica import demodular
from CamadaEnlace import aplicar_enlace_rx

HOST = "127.0.0.1"
PORTA = 65432


@dataclass
class RuntimeRX:
    """Estado compartilhado do receptor enquanto o Streamlit recarrega."""
    fila_resultados: queue.Queue = field(default_factory=queue.Queue)
    receptor_iniciado: bool = False
    erro_receptor: str = ""
    status: str = "Receptor ainda não iniciado."
    lock: threading.Lock = field(default_factory=threading.Lock)


def receber_tudo(conexao: socket.socket, quantidade_bytes: int) -> bytes:
    """Lê exatamente a quantidade de bytes informada do socket, repetindo recv até
    completar a mensagem.
    """
    dados = b""
    while len(dados) < quantidade_bytes:
        pacote = conexao.recv(quantidade_bytes - len(dados))
        if not pacote:
            raise ConnectionError("Conexão encerrada antes do fim da mensagem.")
        dados += pacote
    return dados


def receber_objeto(conexao: socket.socket) -> Any:
    """Recebe um objeto serializado pelo socket, lendo primeiro o tamanho e depois os bytes
    do conteúdo.
    """
    cabecalho = receber_tudo(conexao, 4)
    tamanho = struct.unpack("!I", cabecalho)[0]
    dados_serializados = receber_tudo(conexao, tamanho)
    return pickle.loads(dados_serializados)


def enviar_objeto(objeto: Any) -> None:
    """Serializa um objeto Python, envia seu tamanho e depois o conteúdo por socket TCP
    para o receptor.
    """
    dados = pickle.dumps(objeto)
    cabecalho = struct.pack("!I", len(dados))
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as cliente:
        cliente.connect((HOST, PORTA))
        cliente.sendall(cabecalho + dados)


def tratar_cliente(conexao: socket.socket, endereco, runtime: RuntimeRX) -> None:
    """Trata uma conexão recebida pelo RX: recebe pacote, demodula, aplica enlace no
    receptor e registra o resultado.
    """
    try:
        pacote = receber_objeto(conexao)
        sinal_recebido = np.array(pacote["sinal_recebido"])
        tempo = np.array(pacote["tempo"], dtype=float)
        tipo_modulacao = pacote["tipo_modulacao"]
        tecnica = pacote["tecnica"]
        metodo_enlace = pacote["metodo_enlace"]
        quantidade_bits_fisica = int(pacote["quantidade_bits_fisica"])
        quantidade_bits_dados_original = int(pacote["quantidade_bits_dados_original"])

        bits_demodulados = demodular(sinal_recebido, tempo, tipo_modulacao, tecnica, quantidade_bits_fisica)
        resultado_enlace = aplicar_enlace_rx(bits_demodulados, metodo_enlace, quantidade_bits_dados_original)

        runtime.fila_resultados.put({
            "horario": time.strftime("%H:%M:%S"),
            "endereco": str(endereco),
            "tipo_entrada": pacote.get("tipo_entrada", "Bits"),
            "tipo_modulacao": tipo_modulacao,
            "tecnica": tecnica,
            "metodo_enlace": metodo_enlace,
            "sigma": float(pacote["sigma"]),
            "bits_demodulados": bits_demodulados,
            **resultado_enlace,
        })
    except Exception as erro:
        runtime.fila_resultados.put({
            "horario": time.strftime("%H:%M:%S"),
            "erro": f"Erro ao tratar conexão de {endereco}: {erro}",
        })
    finally:
        conexao.close()


def loop_servidor(runtime: RuntimeRX) -> None:
    """Mantém o socket servidor do RX aberto, aceitando conexões e criando uma thread para
    cada cliente recebido.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as servidor:
            servidor.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            servidor.bind((HOST, PORTA))
            servidor.listen()
            with runtime.lock:
                runtime.status = f"RX aguardando transmissões em {HOST}:{PORTA}."
            while True:
                conexao, endereco = servidor.accept()
                threading.Thread(target=tratar_cliente, args=(conexao, endereco, runtime), daemon=True).start()
    except OSError as erro:
        with runtime.lock:
            runtime.erro_receptor = f"Não foi possível abrir {HOST}:{PORTA}. Feche outro receptor nessa porta. Detalhe: {erro}"
            runtime.status = "Erro ao iniciar receptor."
    except Exception as erro:
        with runtime.lock:
            runtime.erro_receptor = f"Erro inesperado no receptor: {erro}"
            runtime.status = "Erro inesperado no receptor."


def iniciar_receptor(runtime: RuntimeRX) -> None:
    """Inicia a thread principal do receptor, garantindo que ela seja criada apenas uma
    vez.
    """
    with runtime.lock:
        if runtime.receptor_iniciado:
            return
        runtime.receptor_iniciado = True
        runtime.status = "Iniciando receptor em thread..."
    threading.Thread(target=loop_servidor, args=(runtime,), daemon=True).start()
