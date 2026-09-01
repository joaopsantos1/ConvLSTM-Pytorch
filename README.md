# ConvLSTM para previsão de radar

Este projeto treina uma ConvLSTM para nowcasting: a partir de 5 mapas de radar
observados, o modelo prevê os 6 mapas seguintes. Ele suporta os conjuntos
TAASRAD e MeteoNet já incluídos em `data/`.

O ponto de entrada é `train.py`. Veja o guia completo em
[RUN_GUIDE.md](RUN_GUIDE.md).

## Fluxo do projeto

```text
arquivos .npy -> datasets.py -> modelo ConvLSTM -> train.py -> checkpoints/ e results/
```

- `config.py`: caminhos dos arquivos e hiperparâmetros.
- `datasets.py`: leitura em modo memória mapeada, normalização e divisão de
  cada amostra em entrada/alvo.
- `convlstm.py`: implementação da célula recorrente convolucional.
- `model.py`: encoder que lê a sequência observada e forecaster que cria a
  sequência futura.
- `train.py`: treinamento, early stopping, teste e salvamento dos resultados.
- `utils.py`: seeds, métricas e serialização JSON.

## Dados esperados

Cada arquivo `.npy` deve ter formato:

```text
(amostras, passos_temporais, canais, altura, largura)
```

Os arquivos presentes têm formato `(N, 11, 1, 120, 120)`. Com a configuração
padrão, os passos `0..4` são a entrada e `5..10` são o alvo. Não há janela
deslizante adicional: cada linha do arquivo já representa uma sequência pronta.

## Execução rápida

No PowerShell, com o ambiente virtual ativado:

```powershell
python train.py --source taasrad
```

ou:

```powershell
python train.py --source meteonet
```

Por padrão são executadas cinco seeds. Ajuste `seeds`, `epochs` ou
`batch_size` em `config.py` para uma execução menor de teste.
