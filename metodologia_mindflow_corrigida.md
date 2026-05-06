# Metodologia — MindFlow AI
## Pipeline de Classificação de Engajamento (versão corrigida)

---

## Visão geral do fluxo

```
DAiSEE
  │
  ├──► FACE (MediaPipe)  ──► Feature Engineering facial  ──┐
  ├──► POSE (MediaPipe)  ──► Feature Engineering postural ──┼──► Early Fusion
  └──► IRIS (MediaPipe)  ──► Feature Engineering ocular  ──┘        │
                                                               Normalização
                                                                     │
                                                                   SMOTE
                                                                     │
                                                                PCA (opcional)
                                                                     │
                                                          Divisão por sujeito
                                                         ┌──────────┼──────────┐
                                                    20% Valid.  60% Treino  20% Teste
                                                         └──────────┼──────────┘
                                                                     │
                                                                   LSTM
                                                             (+ 1D-CNN opcional)
```

---

## 1. Entrada e extração de features (MediaPipe)

**Dataset:** DAiSEE

O DAiSEE é um dataset público de vídeos de estudantes em sessões de aprendizado online, anotado com quatro níveis de engajamento, tédio, confusão e frustração. A captura das features é feita pelo MediaPipe, que processa cada frame e extrai coordenadas espaciais de três modalidades em paralelo, sem armazenar o vídeo original — apenas os pontos matemáticos resultantes.

| Modalidade | Saída do MediaPipe |
|---|---|
| **Face** | 468 landmarks faciais (mesh 3D) |
| **Pose** | 33 pontos corporais (esqueleto) |
| **Iris** | Vetor de direção do olhar |

> **Conformidade LGPD:** os frames originais são descartados imediatamente após a extração dos landmarks. O sistema processa apenas coordenadas numéricas, sem reconhecimento facial ou armazenamento de imagem.

---

## 2. Feature Engineering (etapa explicitada)

Antes de qualquer fusão, cada modalidade passa por uma etapa de engenharia de features para transformar os landmarks brutos em representações semanticamente relevantes para a detecção de engajamento.

**Features faciais (a partir dos 468 landmarks):**
- Action Units (AUs) estimadas: sobrancelhas franzidas, cantos da boca, fechamento dos olhos
- Razão aspecto dos olhos (EAR — Eye Aspect Ratio) para detecção de sonolência
- Distância entre pontos-chave para expressões de confusão e tédio

**Features posturais (a partir dos 33 pontos de pose):**
- Ângulo de inclinação do tronco (engajamento vs. recuo)
- Ângulo cervical (cabeça para frente = atenção, para trás = desengajamento)
- Distância ombro–orelha como proxy de tensão postural

**Features oculares (a partir do vetor do iris):**
- Estabilidade do ponto de foco ao longo de uma janela de 30 frames
- Desvio padrão do vetor de olhar (alto desvio = mente dispersa)
- Velocidade de sacada ocular

---

## 3. Pré-processamento

A ordem das etapas de pré-processamento segue uma sequência lógica que garante coerência estatística das amostras sintéticas geradas.

### 3.1 Early Fusion

As features das três modalidades são concatenadas em um único vetor de representação unificada. A fusão precoce (*early fusion*) é adotada pela sua simplicidade e por ser adequada quando as modalidades são complementares e capturadas de forma sincronizada.

```
[feat_facial | feat_postural | feat_ocular]  →  vetor ~120 dimensões
```

> **Nota:** a alternativa de *late fusion* (treinar modelos separados por modalidade e combinar as predições) pode oferecer desempenho superior e é indicada como trabalho futuro.

### 3.2 Normalização

Antes da síntese de amostras, o vetor unificado é normalizado para que todas as features contribuam de forma equilibrada ao modelo, independentemente da escala original de cada modalidade.

- Método: **Z-score** (média zero, desvio padrão unitário) por feature
- Aplicado após a fusão, sobre o vetor completo

### 3.3 Balanceamento — SMOTE

O DAiSEE é desbalanceado, com predominância da classe "engajado" em relação às classes "entediado" e "confuso". O SMOTE (*Synthetic Minority Oversampling Technique*) é aplicado **após a fusão e a normalização**, sobre o vetor unificado de ~120 dimensões.

> **Correção crítica:** aplicar SMOTE sobre as modalidades separadas (antes da fusão) geraria amostras sintéticas sem correspondência entre os canais — um rosto sintético não estaria vinculado a uma postura sintética coerente. Ao operar sobre o vetor já fundido, todas as dimensões de uma amostra sintética pertencem ao mesmo "sujeito imaginário".

### 3.4 Redução de dimensionalidade — PCA

O PCA (*Principal Component Analysis*) é aplicado como etapa opcional de redução de dimensionalidade, retendo os componentes que explicam 95% da variância dos dados.

- A necessidade do PCA deve ser validada empiricamente: comparar o desempenho do modelo com e sem redução
- Com vetores de ~120 dimensões, o PCA pode não ser imprescindível — LSTMs lidam bem com essa escala

---

## 4. Divisão dos dados

A divisão é feita **por sujeito** (*subject-independent split*), garantindo que nenhum sujeito apareça simultaneamente em partições diferentes.

| Partição | Proporção | Uso |
|---|---|---|
| Treino | 60% | Ajuste dos pesos do modelo |
| Validação | 20% | Seleção de hiperparâmetros e early stopping |
| Teste | 20% | Avaliação final, executada uma única vez |

> **Correção crítica — data leakage:** uma divisão aleatória por clip (e não por sujeito) permitiria que o modelo "visse" rostos do mesmo sujeito no treino e no teste, inflando artificialmente as métricas. A divisão por sujeito é o padrão correto para datasets de vídeo com múltiplos clips por participante.

---

## 5. Modelo

### LSTM (Long Short-Term Memory)

A entrada do modelo é uma **sequência temporal de vetores de features**, representando o estado do sujeito ao longo de uma janela de frames (ex.: 30 frames ≈ 1 segundo a 30fps). O LSTM é a arquitetura indicada para essa estrutura de dados, pois captura dependências temporais — a evolução do estado de engajamento ao longo do tempo.

```
Entrada: sequência [t-29, t-28, ..., t] de vetores de features
Saída:   {engajamento, tédio, confusão} + confiança
```

**Arquitetura sugerida:**

```
LSTM (128 unidades) → Dropout (0.3) → LSTM (64 unidades) → Dense (3) → Softmax
```

### 1D-CNN como complemento (opcional)

Uma camada de convolução 1D pode ser adicionada antes do LSTM para extrair padrões locais na sequência temporal antes do processamento recorrente, formando uma arquitetura híbrida CNN+LSTM.

> **Alinhamento com o documento:** o diagrama original indicava CNN como modelo principal, enquanto o documento descrevia LSTM. Esta versão corrigida adota LSTM como arquitetura primária, alinhando diagrama e texto, com CNN como camada auxiliar opcional.

---

## Resumo das correções aplicadas

| # | Ponto corrigido | Original | Corrigido |
|---|---|---|---|
| 1 | Feature engineering | Não explicitada | Etapa dedicada por modalidade |
| 2 | Posição do SMOTE | Antes da fusão (por modalidade) | Após fusão e normalização |
| 3 | Ordem norm. vs. SMOTE | Indefinida | Normalização → SMOTE |
| 4 | Divisão dos dados | Por clip (risco de leakage) | Por sujeito (subject-independent) |
| 5 | Modelo | CNN (contradição com o doc.) | LSTM principal + 1D-CNN opcional |
| 6 | PCA vs. SVD | Indecisão entre os dois | PCA com validação empírica |
