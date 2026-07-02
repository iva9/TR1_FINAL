"""Camada de Enlace do simulador de TR1.

Este arquivo concentra conversões, enquadramento, detecção de erros e correção
por Hamming. A interface e o simulador chamam estas funções para manter a
camada de enlace separada como solicitado no enunciado.
"""

# Flag padrão usada para delimitar quadros no enquadramento por bits.
FLAG_BITS = "01111110"
# Flag em formato de byte usada no enquadramento por bytes/caracteres.
FLAG_BYTE = 0x7E
# Byte de escape usado para proteger flags que aparecem dentro dos dados.
ESC_BYTE = 0x7D
# Polinômio refletido do CRC-32 IEEE.
CRC32_POLINOMIO = 0xEDB88320


def texto_para_bits(texto: str) -> str:
    """Converte uma string de texto em uma sequência binária usando codificação UTF-8, com
    8 bits para cada byte.
    """
    return "".join(format(byte, "08b") for byte in texto.encode("utf-8"))


def bits_para_bytes(bits: str) -> bytes:
    """Converte uma sequência de bits em bytes; completa com zeros à direita quando
    necessário para fechar bytes de 8 bits.
    """
    if not bits:
        return b""
    resto = len(bits) % 8
    if resto:
        bits += "0" * (8 - resto)
    return bytes(int(bits[i:i + 8], 2) for i in range(0, len(bits), 8))


def bytes_para_bits(dados: bytes) -> str:
    """Converte bytes em uma string de bits, representando cada byte com exatamente 8 bits."""
    return "".join(format(byte, "08b") for byte in dados)


def bits_para_texto(bits: str) -> str:
    """Tenta reconstruir texto UTF-8 a partir de uma sequência de bits; retorna vazio se os
    bits não formarem texto válido.
    """
    if len(bits) % 8 != 0:
        return ""
    try:
        return bits_para_bytes(bits).decode("utf-8")
    except UnicodeDecodeError:
        return ""


def bits_validos(bits: str) -> bool:
    """Verifica se uma string representa uma sequência binária não vazia, contendo apenas
    os caracteres 0 e 1.
    """
    return bool(bits) and all(bit in "01" for bit in bits)


def validar_entrada(tipo_entrada: str, entrada: str) -> str:
    """Recebe texto ou bits da interface e retorna bits de dados."""
    if tipo_entrada == "Bits":
        bits = entrada.strip().replace(" ", "")
        if not bits_validos(bits):
            raise ValueError("Entrada binária inválida. Use apenas 0 e 1.")
        return bits
    if not entrada:
        raise ValueError("Digite um texto para transmitir.")
    return texto_para_bits(entrada)


def contagem_de_caracteres_codificar(dados: bytes) -> bytes:
    """Aplica enquadramento por contagem de caracteres, colocando no primeiro byte o
    tamanho total do quadro.
    """
    tamanho_total = len(dados) + 1
    if tamanho_total > 255:
        raise ValueError("A contagem usa 1 byte. Use no máximo 254 bytes de dados.")
    return bytes([tamanho_total]) + dados


def contagem_de_caracteres_decodificar(quadro: bytes) -> bytes:
    """Remove o enquadramento por contagem de caracteres, lendo o tamanho no primeiro byte
    e extraindo os dados.
    """
    if not quadro:
        raise ValueError("Quadro vazio.")
    tamanho_total = quadro[0]
    if tamanho_total > len(quadro):
        raise ValueError("Quadro inválido: tamanho informado maior que o quadro recebido.")
    return quadro[1:tamanho_total]


def insercao_de_bytes_com_flags(dados: bytes) -> bytes:
    """Aplica enquadramento com flags em bytes, usando 0x7E como delimitador e 0x7D como
    escape.
    """
    quadro = bytearray([FLAG_BYTE])
    for byte in dados:
        if byte in (FLAG_BYTE, ESC_BYTE):
            quadro.append(ESC_BYTE)
            quadro.append(byte ^ 0x20)
        else:
            quadro.append(byte)
    quadro.append(FLAG_BYTE)
    return bytes(quadro)


def remocao_de_bytes_com_flags(quadro: bytes) -> bytes:
    """Remove flags e desfaz escapes do enquadramento por bytes, recuperando os dados
    originais.
    """
    if len(quadro) < 2 or quadro[0] != FLAG_BYTE or quadro[-1] != FLAG_BYTE:
        raise ValueError("Quadro inválido: flags de início/fim não encontradas.")
    conteudo = quadro[1:-1]
    dados = bytearray()
    indice = 0
    while indice < len(conteudo):
        byte = conteudo[indice]
        if byte == ESC_BYTE:
            if indice + 1 >= len(conteudo):
                raise ValueError("Quadro inválido: escape sem byte seguinte.")
            dados.append(conteudo[indice + 1] ^ 0x20)
            indice += 2
        else:
            dados.append(byte)
            indice += 1
    return bytes(dados)


def insercao_de_bits_com_flags(bits: str) -> str:
    """Aplica bit stuffing: insere 0 após cinco bits 1 consecutivos e adiciona flags no
    início e no fim.
    """
    resultado = ""
    uns_consecutivos = 0
    for bit in bits:
        resultado += bit
        if bit == "1":
            uns_consecutivos += 1
            if uns_consecutivos == 5:
                resultado += "0"
                uns_consecutivos = 0
        else:
            uns_consecutivos = 0
    return FLAG_BITS + resultado + FLAG_BITS


def remocao_de_bits_com_flags(quadro: str) -> str:
    """Remove flags e desfaz o bit stuffing, retirando zeros inseridos após cinco bits 1
    consecutivos.
    """
    if not (quadro.startswith(FLAG_BITS) and quadro.endswith(FLAG_BITS)):
        raise ValueError("Quadro inválido: flags de início/fim não encontradas.")
    conteudo = quadro[len(FLAG_BITS):-len(FLAG_BITS)]
    resultado = ""
    uns_consecutivos = 0
    indice = 0
    while indice < len(conteudo):
        bit = conteudo[indice]
        resultado += bit
        if bit == "1":
            uns_consecutivos += 1
            if uns_consecutivos == 5:
                proximo = indice + 1
                if proximo < len(conteudo) and conteudo[proximo] == "0":
                    indice += 1
                uns_consecutivos = 0
        else:
            uns_consecutivos = 0
        indice += 1
    return resultado


def adicionar_paridade_par(bits: str) -> str:
    """Adiciona um bit de paridade para que a quantidade total de bits 1 fique par."""
    bit_paridade = "0" if bits.count("1") % 2 == 0 else "1"
    return bits + bit_paridade


def verificar_paridade_par(bits_com_paridade: str) -> bool:
    """Verifica a paridade par contando os bits 1; retorna verdadeiro quando a quantidade
    total é par.
    """
    return bits_validos(bits_com_paridade) and bits_com_paridade.count("1") % 2 == 0


def calcular_checksum(dados: bytes) -> int:
    """Calcula checksum de 16 bits com soma em complemento de 1 sobre palavras de 16 bits."""
    if len(dados) % 2 != 0:
        dados += b"\x00"
    soma = 0
    for indice in range(0, len(dados), 2):
        palavra = (dados[indice] << 8) + dados[indice + 1]
        soma += palavra
        soma = (soma & 0xFFFF) + (soma >> 16)
    return (~soma) & 0xFFFF


def anexar_checksum(bits: str) -> str:
    """Calcula o checksum dos dados e anexa seus 16 bits ao final da mensagem."""
    checksum = calcular_checksum(bits_para_bytes(bits))
    return bits + format(checksum, "016b")


def verificar_checksum(bits_com_checksum: str) -> bool:
    """Separa os dados do checksum recebido, recalcula o checksum e compara os dois
    valores.
    """
    if len(bits_com_checksum) < 16:
        return False
    dados = bits_com_checksum[:-16]
    recebido = bits_com_checksum[-16:]
    calculado = format(calcular_checksum(bits_para_bytes(dados)), "016b")
    return recebido == calculado


def calcular_crc32(dados: bytes) -> int:
    """Calcula o CRC32".
    """
    crc = 0xFFFFFFFF
    for byte in dados:
        crc ^= byte
        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ CRC32_POLINOMIO
            else:
                crc >>= 1
    return crc ^ 0xFFFFFFFF


def anexar_crc32(bits: str) -> str:
    """Calcula o CRC-32 dos dados e anexa seus 32 bits ao final da mensagem."""
    crc = calcular_crc32(bits_para_bytes(bits))
    return bits + format(crc, "032b")


def verificar_crc32(bits_com_crc: str) -> bool:
    """Separa dados e CRC recebido, recalcula o CRC-32 e verifica se os valores coincidem."""
    if len(bits_com_crc) < 32:
        return False
    dados = bits_com_crc[:-32]
    recebido = bits_com_crc[-32:]
    calculado = format(calcular_crc32(bits_para_bytes(dados)), "032b")
    return recebido == calculado


def quantidade_bits_paridade(quantidade_dados: int) -> int:
    """Calcula quantos bits de paridade são necessários para Hamming usando a condição 2^r
    >= k + r + 1.
    """
    r = 0
    while 2 ** r < quantidade_dados + r + 1:
        r += 1
    return r


def eh_potencia_de_2(numero: int) -> bool:
    """Verifica se um número é potência de 2, critério usado para identificar posições de
    paridade no Hamming.
    """
    return numero > 0 and (numero & (numero - 1)) == 0


def hamming_codificar(bits_dados: str) -> str:
    """Codifica os bits usando Hamming genérico, colocando paridades nas posições 1, 2, 4,
    8 e assim por diante.
    """
    quantidade_dados = len(bits_dados)
    quantidade_paridade = quantidade_bits_paridade(quantidade_dados)
    tamanho_total = quantidade_dados + quantidade_paridade
    codigo = ["0"] * (tamanho_total + 1)
    indice_dados = 0
    for posicao in range(1, tamanho_total + 1):
        if not eh_potencia_de_2(posicao):
            codigo[posicao] = bits_dados[indice_dados]
            indice_dados += 1
    for posicao_paridade in range(1, tamanho_total + 1):
        if eh_potencia_de_2(posicao_paridade):
            paridade = 0
            for posicao in range(1, tamanho_total + 1):
                if posicao & posicao_paridade:
                    paridade ^= int(codigo[posicao])
            codigo[posicao_paridade] = str(paridade)
    return "".join(codigo[1:])


def hamming_decodificar(codigo_recebido: str):
    """Decodifica Hamming, calcula a síndrome, corrige erro de 1 bit quando possível e
    remove os bits de paridade.
    """
    codigo = ["0"] + list(codigo_recebido)
    tamanho_total = len(codigo_recebido)
    sindrome = 0
    for posicao_paridade in range(1, tamanho_total + 1):
        if eh_potencia_de_2(posicao_paridade):
            paridade = 0
            for posicao in range(1, tamanho_total + 1):
                if posicao & posicao_paridade:
                    paridade ^= int(codigo[posicao])
            if paridade != 0:
                sindrome += posicao_paridade
    if 0 < sindrome <= tamanho_total:
        codigo[sindrome] = "1" if codigo[sindrome] == "0" else "0"
    dados = ""
    for posicao in range(1, tamanho_total + 1):
        if not eh_potencia_de_2(posicao):
            dados += codigo[posicao]
    return dados, sindrome, "".join(codigo[1:])


def aplicar_enlace_tx(bits_dados: str, metodo_enlace: str) -> tuple[str, dict]:
    """Aplica no TX o método de enlace escolhido e retorna os bits que irão para a física."""
    info = {"metodo": metodo_enlace}
    dados = bits_para_bytes(bits_dados)

    if metodo_enlace == "Nenhum":
        bits_tx = bits_dados
    elif metodo_enlace == "Contagem de Caracteres":
        quadro = contagem_de_caracteres_codificar(dados)
        bits_tx = bytes_para_bits(quadro)
        info["quadro"] = bits_tx
    elif metodo_enlace == "Inserção de Bytes/Caracteres com Flags":
        quadro = insercao_de_bytes_com_flags(dados)
        bits_tx = bytes_para_bits(quadro)
        info["quadro"] = bits_tx
    elif metodo_enlace == "Inserção de Bits com Flags":
        bits_tx = insercao_de_bits_com_flags(bits_dados)
        info["quadro"] = bits_tx
    elif metodo_enlace == "Paridade Par":
        bits_tx = adicionar_paridade_par(bits_dados)
        info["controle"] = bits_tx[-1]
    elif metodo_enlace == "Checksum":
        bits_tx = anexar_checksum(bits_dados)
        info["controle"] = bits_tx[-16:]
    elif metodo_enlace == "CRC-32":
        bits_tx = anexar_crc32(bits_dados)
        info["controle"] = bits_tx[-32:]
    elif metodo_enlace == "Hamming":
        bits_tx = hamming_codificar(bits_dados)
        info["codigo_hamming"] = bits_tx
    else:
        raise ValueError("Método de enlace desconhecido.")

    info["bits_enviados_fisica"] = bits_tx
    return bits_tx, info


def aplicar_enlace_rx(bits_recebidos: str, metodo_enlace: str, quantidade_bits_dados_original: int) -> dict:
    """Executa no RX o desenquadramento, verificação ou correção do método escolhido."""
    resultado = {
        "metodo": metodo_enlace,
        "bits_recebidos_enlace": bits_recebidos,
        "valido": True,
        "mensagem_status": "Mensagem processada.",
        "bits_dados_recuperados": bits_recebidos,
    }

    try:
        if metodo_enlace == "Nenhum":
            resultado["bits_dados_recuperados"] = bits_recebidos[:quantidade_bits_dados_original]
        elif metodo_enlace == "Contagem de Caracteres":
            recuperado = contagem_de_caracteres_decodificar(bits_para_bytes(bits_recebidos))
            resultado["bits_dados_recuperados"] = bytes_para_bits(recuperado)[:quantidade_bits_dados_original]
        elif metodo_enlace == "Inserção de Bytes/Caracteres com Flags":
            recuperado = remocao_de_bytes_com_flags(bits_para_bytes(bits_recebidos))
            resultado["bits_dados_recuperados"] = bytes_para_bits(recuperado)[:quantidade_bits_dados_original]
        elif metodo_enlace == "Inserção de Bits com Flags":
            recuperado = remocao_de_bits_com_flags(bits_recebidos)
            resultado["bits_dados_recuperados"] = recuperado[:quantidade_bits_dados_original]
        elif metodo_enlace == "Paridade Par":
            resultado["valido"] = verificar_paridade_par(bits_recebidos)
            resultado["bits_dados_recuperados"] = bits_recebidos[:-1]
            resultado["mensagem_status"] = "Paridade válida." if resultado["valido"] else "Erro detectado pela paridade."
        elif metodo_enlace == "Checksum":
            resultado["valido"] = verificar_checksum(bits_recebidos)
            resultado["bits_dados_recuperados"] = bits_recebidos[:-16]
            resultado["checksum_recebido"] = bits_recebidos[-16:] if len(bits_recebidos) >= 16 else ""
            resultado["mensagem_status"] = "Checksum válido." if resultado["valido"] else "Erro detectado pelo checksum."
        elif metodo_enlace == "CRC-32":
            resultado["valido"] = verificar_crc32(bits_recebidos)
            resultado["bits_dados_recuperados"] = bits_recebidos[:-32]
            resultado["crc_recebido"] = bits_recebidos[-32:] if len(bits_recebidos) >= 32 else ""
            resultado["mensagem_status"] = "CRC-32 válido." if resultado["valido"] else "Erro detectado pelo CRC-32."
        elif metodo_enlace == "Hamming":
            dados, sindrome, codigo_corrigido = hamming_decodificar(bits_recebidos)
            resultado["bits_dados_recuperados"] = dados[:quantidade_bits_dados_original]
            resultado["sindrome"] = sindrome
            resultado["codigo_corrigido"] = codigo_corrigido
            resultado["mensagem_status"] = "Sem erro detectado pelo Hamming." if sindrome == 0 else f"Hamming corrigiu erro na posição {sindrome}."
        else:
            raise ValueError("Método de enlace desconhecido.")
    except Exception as erro:
        resultado["valido"] = False
        resultado["mensagem_status"] = f"Erro ao processar enlace no RX: {erro}"

    resultado["texto_recuperado"] = bits_para_texto(resultado["bits_dados_recuperados"])
    return resultado
