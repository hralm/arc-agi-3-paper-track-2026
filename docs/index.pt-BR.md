---
layout: default
title: "Como o trabalho foi feito"
description: "ARC Prize 2026 Paper Track: o relato do autor sobre as seis semanas; também em inglês"
lang: pt-BR
---

[English](./) · Português (Brasil)

# Certificando a ausência de efeito e identificando a condição de vitória sob pontuação por ação no ARC-AGI-3

*ARC Prize 2026 · Paper Track · como o trabalho foi feito, pelo autor*

- Artigo (PDF): [`paper/`](https://github.com/hralm/arc-agi-3-paper-track-2026/tree/main/paper), acrescentado na submissão, em 8 de novembro de 2026
- Código, bancada, testes, manifestos, os cinco notebooks e os arquivos por trás de cada rodada da fase A: [este repositório](https://github.com/hralm/arc-agi-3-paper-track-2026) (`notebooks/`, `bench/results/phase-a/`, `bench/phase_a.py`)
- Notebooks: a [entrada elegível](https://www.kaggle.com/code/hralmjc/what-exploration-can-ignore-duck-fork-k-playing) (*fork* do Duck, K jogando) e os quatro sobre a base dfranzen: [controle](https://www.kaggle.com/code/hralmjc/what-exploration-can-ignore-dfranzen-control), [K jogando](https://www.kaggle.com/code/hralmjc/what-exploration-can-ignore-dfranzen-k-playing), [K aconselhando](https://www.kaggle.com/code/hralmjc/what-exploration-can-ignore-dfranzen-k-advising), [último recurso](https://www.kaggle.com/code/hralmjc/what-exploration-can-ignore-dfranzen-last-resort)

## A pergunta de que parti

O ARC-AGI-3 diz avaliar a definição de objetivos. Mas todo jogo já tem uma condição de vitória, e a pontuação cobra cada
ação que você toma. Então, o que um agente pode fazer aqui? Pode identificar o objetivo que o jogo fixou, e não escolher
um. Eu queria saber se esse passo, *identificar a condição de vitória*, pode ser dado sem gastar as ações que a pontuação
conta, e se ele chega sequer a alcançar a pontuação.

Para medi-lo, construí um explorador de grafo (K). Ele emite certificados revogáveis de ausência de efeito para ações que
não mudam nada, sinaliza estado oculto quando um certificado falha ou uma ação segura mata, e ordena hipóteses de objetivo
que apenas ordenam as ações candidatas. Depois o pus dentro do *harness* de um agente de modelo de linguagem de pesos
abertos, de três maneiras: jogando, aconselhando e como último recurso. O artigo é o relato do que essa camada afirmou, do
uso que pude observar e do que as comparações de pontuação podem e não podem estabelecer. A maior parte do que elas
estabelecem é negativa, e esse é o resultado: em 50 de 66 vitórias locais o explorador venceu sem que o identificador
nomeasse o objetivo, e nenhum modo produziu ganho mensurável de pontuação.

## Tempo, máquina e orçamento

Seis semanas, de 29 de setembro a 8 de novembro de 2026. Um autor, com agentes de modelos de linguagem fazendo o trabalho
executivo (código, testes, bancadas, rascunhos) sob regras que fixei e revisei, como declara a Declaração do artigo.

Minha máquina é um pequeno PC doméstico, sem GPU dedicada. Esse fato, sozinho, moldou o projeto. Tudo o que precisava de
GPU rodou dentro da cota semanal do Kaggle, de modo que cada rodada da fase A disputou com todas as outras as mesmas
30 horas, e a bancada local tinha de ser algo que uma CPU terminasse de um dia para o outro: 25 jogos públicos,
4 sementes, 1.500 passos, cerca de 35 minutos por braço. Conjecturo que a maioria das entradas de ponta adaptou o
próprio modelo, com hardware além do que o Kaggle concede. Não pude testar isso, e não tentei. A pergunta que eu podia
testar era mais estreita: uma camada que *sabe coisas sobre o jogo* ajuda o modelo que o joga?

## Como o trabalho foi organizado

O agente acoplado passou por duas linhas. A linha Duck (v1 a v7) roda sobre o *harness* da Tufa Labs, e a sua v5 é a
entrada elegível. A linha dfranzen (v8 a v14.1) roda sobre o notebook aberto do Milestone 2. A sua última versão terminou
a reexecução no conjunto oculto depois do prazo e é relatada apenas como medida.

A partir de 27 de outubro, antes da primeira versão acoplada, toda versão seguiu um só protocolo. Invariantes e
afirmações eram escritos primeiro. Um modelo construía a versão e os seus testes. Um modelo diferente, numa sessão nova,
atacava o artefato congelado contra o seu manifesto de hashes e devolvia um veredito com uma prova reproduzível para cada
achado. Todo defeito voltava com um teste de regressão e um mutante, uma reintrodução deliberada do defeito que o teste
novo tem de apanhar. Na v14.1 eram 92 testes e 85 mutantes. Não submeti nenhuma versão com um achado bloqueante em
aberto.

As medições seguiram a mesma regra: decidir antes do número. A regra de leitura da fase A foi fixada depois de uma rodada
piloto e antes das demais: um braço mostra efeito só se a sua média difere da média da base por mais de duas vezes o maior
DP amostral do par. Se uma rodada seria lida também foi decidido antes do seu resultado, nos dois sentidos. Uma rodada de
controle que teve de ser reexecutada porque o texto dos notebooks mudou foi lida, pois o lote ainda estava aberto. Uma
rodada de último recurso com nove exceções vindas da minha camada foi excluída e substituída, pela regra escrita dias
antes. As fontes foram lidas por inteiro antes de serem citadas, e o texto final foi revisto bloco a bloco, primeiro a
fidelidade, depois o estilo.

## O que deu errado

- **A primeira submissão pontuada ficou em 0,75.** K tomou a maior parte dos turnos do modelo, e a pontuação cobra cada
  um deles. Mantive o número no artigo como evidência compatível com o custo de ações do explorador, e construí os modos
  aconselhando e de último recurso para limitar a intervenção no tempo e nas ações. O teto limita a intervenção, e não a
  perda.
- **O servidor do modelo levou de 305 a 1.494 segundos para carregar,** e em quatro de oito execuções os jogos foram
  liberados enquanto ele ainda carregava. Uma rodada de controle foi perdida porque o servidor nunca ficou pronto. Foi
  reexecutada.
- **A última submissão não teve o orçamento da sua reexecução.** A v14.1 entrou em 2 de novembro, às 22:30 UTC. A sua
  reexecução de nove horas terminou depois do prazo, logo não pôde ser selecionada. Outra equipe sobre a mesma base relata
  a mesma perda. A lição é aritmética: uma última submissão precisa do tempo inteiro de reexecução, mais a margem da fila,
  antes do prazo de seleção, e eu não o tinha.
- **O Kaggle montou as entradas dos notebooks em dois leiautes diferentes.** Três envios de um notebook morreram antes de
  qualquer jogo rodar. Os notebooks públicos agora resolvem os dois leiautes e registram qual deles encontraram.
- **No único jogo que diagnostiquei, o modelo nunca mencionou as notas do conselheiro** no seu raciocínio visível. As
  suas jogadas caras vieram de um laço de código que ele mesmo escreveu, e que contornava a lista de ações podada. Esse
  rastro não me deixa culpar a camada pela diferença, e também não me deixa absolvê-la.

## O que aprendi

- **Três jogos públicos e três rodadas detectam só efeitos grandes.** No mesmo dia, com o mesmo notebook sem nenhuma
  modificação, um jogo pontuou de 21 a 100. Com DPs amostrais acima de 15, a barra ficou em cerca de 45 pontos, e nenhum
  braço chegou perto dela.
- **Dois lotes da mesma base, com cinco dias de intervalo, diferiram em cerca de 23 pontos** (74,91 contra 52,30). Os
  níveis da fase A não se comparam entre lotes, de modo que toda comparação do artigo é lida dentro do lote em que foi
  rodada.
- **Decidir antes do número é o que torna um resultado negativo digno de publicação.** O que conta como rodada, como um
  efeito é lido, qual braço é descritivo: tudo isso foi escrito antes de os dados chegarem, e é por isso que posso relatar
  0,75 e 50 de 66 sem pestanejar.
- **Os *red teams* executados por outros modelos encontraram uma mesma classe de defeito, repetidas vezes:** lugares onde
  o que o agente afirma, no seu prompt ou no seu plano, diverge do que o ambiente permite. O artigo oferece essa lista como
  uma lista de verificação para o trabalho em *harness*.
- **O explorador vence por busca, e o identificador na maioria das vezes erra.** Se a definição de objetivos deve ser
  pontuada como componente da inteligência, o benchmark teria de pedir ao agente que enuncie a condição de vitória que
  persegue, e pontuar esse enunciado separadamente da eficiência das ações. Escolher objetivos, em vez de identificar o que
  um jogo fixou, está ainda mais longe, e eu não o testei.

## Declaração

Agentes de grandes modelos de linguagem de provedores diferentes e de perfis diferentes, ocidentais e orientais, foram
usados deliberadamente sob um protocolo fixo. Eu fixei as perguntas, as hipóteses e as regras de trabalho. Os agentes
implementaram o código, rodaram os testes locais e redigiram o texto. Eu adjudiquei toda saída, executei todos os
experimentos no Kaggle e assumo a responsabilidade pelo conteúdo. Nenhum modelo de linguagem é autor. A declaração
completa está no artigo.
