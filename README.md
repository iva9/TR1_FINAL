# Simulador de Camada Física e Camada de Enlace - TR1

Projeto de Teleinformática e Redes 1 com arquitetura modular:

```text
Trabalho_TR1/
├── CamadaFisica.py
├── CamadaEnlace.py
├── Interface.py
├── Simulador.py
├── requirements.txt
├── Relatorio_TR1.pdf
└── README.md
```

## Papel de cada arquivo

- `CamadaFisica.py`: contém as funções de modulação, demodulação e ruído gaussiano.
- `CamadaEnlace.py`: contém enquadramento, detecção de erros, correção por Hamming e conversões entre bits, bytes e texto.
- `Simulador.py`: contém socket TCP local, thread do receptor RX e rotina de comunicação TX/RX.
- `Interface.py`: contém a GUI em Streamlit, separando visualmente Transmissor TX, meio de comunicação e Receptor RX.

## Funcionalidades implementadas

### Camada Física

- NRZ-Polar
- Manchester
- Bipolar
- ASK
- FSK
- QPSK
- 16-QAM
- Demodulação correspondente a cada técnica
- Ruído gaussiano `n(0, σ)` aplicado no canal
- Gráficos com grandezas nos eixos: tempo normalizado no eixo x e bits, clock ou tensão/amplitude no eixo y

### Camada de Enlace

- Contagem de caracteres
- Enquadramento com flags e inserção de bytes/caracteres
- Enquadramento com flags e inserção de bits
- Paridade par
- Checksum
- CRC-32 manual
- Hamming

### Simulador integrado

Fluxo implementado:

```text
TX → Camada de Enlace → Camada Física → Ruído gaussiano → Socket TCP local → RX em thread → Demodulação → Enlace RX → Resultado
```

O socket local usa:

```text
127.0.0.1:65432
```

## Instalação

Dentro da pasta do projeto, execute:

```bash
python -m pip install -r requirements.txt
```

ou, no Windows:

```bash
py -m pip install -r requirements.txt
```

## Execução principal

A interface principal é `Interface.py`:

```bash
python -m streamlit run Interface.py
```

ou:

```bash
py -m streamlit run Interface.py
```

## Como testar

1. Clique em **Iniciar receptor RX**.
2. Digite a entrada no Transmissor TX.
3. Escolha texto ou bits.
4. Escolha o método de enlace.
5. Escolha a modulação da camada física.
6. Ajuste o ruído gaussiano `σ [V]`.
7. Clique em **Transmitir pelo socket**.
8. Confira no Receptor RX os bits demodulados, a verificação/correção e o texto recuperado.

## Observação sobre as unidades

Os sinais são modelados como amplitudes/tensões normalizadas. O eixo y do sinal físico representa `V(t) [V]`, e o ruído gaussiano é somado como `n(0, σ)`, com `σ` também em volts normalizados.

Feito por:
Vinicius de Camargo Bandeira
Ivanov Machado dos Santos
João Felipe Silva Pereira
