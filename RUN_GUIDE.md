# Guia de execução

## 1. Pré-requisitos

Use Windows, Python 3.11 e uma GPU NVIDIA com driver atualizado. O projeto foi
testado neste repositório com:

- Python 3.11;
- PyTorch 2.7.0 com CUDA 12.6;
- NumPy 2.4.6;
- TorchMetrics 1.9.0.

O toolkit CUDA instalado no sistema não é necessário para rodar as wheels do
PyTorch. O driver NVIDIA é necessário.

## 2. Criar e ativar o ambiente virtual

Na raiz do repositório, execute:

```powershell
py -3.11 -m venv menv
.\menv\Scripts\Activate.ps1
```

Se o PowerShell bloquear a ativação, execute uma vez:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

## 3. Instalar dependências

Instale a build CUDA do PyTorch e depois as dependências do projeto:

```powershell
python -m pip install --upgrade pip
python -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu126 --extra-index-url https://pypi.org/simple
python -m pip install -r requirements.txt
```

`torchvision` é instalado junto para compatibilidade com as métricas de imagem.
O projeto não usa `torchaudio`.

## 4. Confirmar que a GPU está disponível

```powershell
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

O resultado esperado contém `2.7.0+cu126`, `12.6`, `True` e o nome da GPU.
Se o terceiro valor for `False`, confirme que o terminal mostra `(menv)` e que
o driver NVIDIA está atualizado.

## 5. Conferir os dados

A estrutura esperada é:

```text
data/
├── TAASRAD/
│   ├── taasrad19_2010_2011_2012_2013_2014_2015_2016_120x120.npy
│   ├── taasrad19_2017_120x120.npy
│   └── taasrad19_2018_2019_120x120.npy
└── Meteonet/
    ├── meteonet_NW_2016_2017_120x120_clean.npy
    ├── meteonet_NW_2017_limit_09_120x120_clean.npy
    └── meteonet_NW_2018_120x120_clean.npy
```

Os nomes e caminhos ficam em `DATA_FILES`, no início de `config.py`. Caso
substitua os dados, mantenha o formato `(N, T, C, H, W)` e garanta que
`T >= input_len + output_len`.

## 6. Rodar o treinamento

TAASRAD:

```powershell
python train.py --source taasrad
```

MeteoNet:

```powershell
python train.py --source meteonet
```

O programa imprime o dispositivo, quantidade de amostras, estatísticas de
normalização e a perda de treino/validação de cada época. O treino executa
todas as épocas definidas em `epochs`.

## 7. Ajustar o experimento

Edite `ExperimentConfig` em `config.py`.

| Campo | Efeito |
| --- | --- |
| `input_len`, `output_len` | Número de frames observados e previstos. A soma deve caber em `T`. |
| `hidden_dims` | Canais das camadas ConvLSTM; valores maiores consomem mais VRAM. |
| `batch_size` | Amostras por lote; reduza se faltar memória na GPU. |
| `epochs` | Número total de épocas executadas em cada seed. |
| `learning_rate` | Taxa de aprendizagem do Adam. |
| `seeds` | Repetições independentes; use `(0,)` para testar rapidamente. |
| `device` | Use `"cuda"` para GPU ou `"cpu"` para forçar CPU. |

Para uma primeira execução curta, altere temporariamente para:

```python
epochs: int = 1
seeds: Tuple[int, ...] = (0,)
```

## 8. Entender os arquivos gerados

Após uma execução para `taasrad`, por exemplo:

```text
checkpoints/taasrad/seed_0/best.pt
results/taasrad/test_metrics_seed_0.csv
results/taasrad/execution_metrics_seed_0.csv
results/taasrad/summary_metrics.csv
```

- `best.pt`: pesos da época com menor perda de validação daquela seed.
- `test_metrics_seed_*.csv`: métricas estatísticas finais no conjunto de teste
  e MSEs consolidados de todo o treino; não há métricas salvas por época.
- `execution_metrics_seed_*.csv`: tempo total, parâmetros, VRAM e consumo de
  energia/emissões medidos pelo CodeCarbon (kWh e kg de CO₂).
- `summary_metrics.csv`: média e desvio padrão de todas as métricas entre as seeds.
- `data/norm_stats/*_norm_stats.json`: média e desvio usados para normalizar
  todos os splits de uma fonte.

## Como o modelo funciona

1. O encoder lê os 5 frames observados, atualizando o estado de cada camada
   ConvLSTM.
2. O forecaster recebe o estado final, usa o último frame observado como ponto
   inicial e cria um frame por vez.
3. No treino, teacher forcing pode substituir parte das previsões pelo alvo
   real; essa probabilidade decai até zero ao longo das épocas.
4. A perda é MSE entre a sequência prevista e a sequência-alvo.

A célula usa convoluções, portanto preserva a vizinhança espacial dos mapas.
Os peepholes são pesos por canal e a normalização é `GroupNorm`; ambas as
opções podem ser desativadas em `config.py`.
