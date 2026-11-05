# Mecânica dos Fluidos — Simuladores e Notebooks Interativos (PPGMMat / UFPel)

Repositório com notebooks e simuladores interativos desenvolvidos na disciplina de **Mecânica dos Fluidos** do **Programa de Pós-Graduação em Modelagem Matemática (PPGMMat — UFPel)**.

---

## 🌐 Simulador Interativo em HTML5 — Exemplo 1 da Aula 05 (Navier-Stokes)

👉 **[Acesse o Simulador Online (GitHub Pages)](https://luisin-py.github.io/mecanica-dos-fluidos/)** *(ou abra [`simulador_exemplo1_aula05.html`](./simulador_exemplo1_aula05.html) / [`index.html`](./index.html))*

### Equação Simulada (Escoamento de Couette-Poiseuille entre Placas Planas Inclinadas)
Escoamento laminar, incompressível e permanente entre duas placas planas paralelas separadas por $h = 2a$ ($y \in [-a, +a]$) e inclinadas de um ângulo $\theta$ com a horizontal, com placa inferior fixa ($u(-a)=0$) e placa superior móvel ($u(+a)=U$):

$$
u(y) = -\frac{K a^2}{2\mu}\left[1 - \frac{y^2}{a^2}\right] + \frac{U}{2}\left[1 + \frac{y}{a}\right], \qquad K = \rho g \sin\theta + \frac{\partial p}{\partial x}
$$

- **Vazão volumétrica**: $Q = \left(-\frac{2}{3}\frac{K a^2}{\mu} + U\right) a b$
- **Velocidade média**: $u_m = -\frac{1}{3}\frac{K a^2}{\mu} + \frac{1}{2}U$
- **Tensão de cisalhamento**: $\tau(y) = K y + \mu\frac{U}{2a}$

---

## 📂 Estrutura dos Arquivos

- [`simulacoes_aula05.ipynb`](./simulacoes_aula05.ipynb) — Notebook Jupyter voltado para o simulador interativo em HTML5 do Exemplo 1 da Aula 05.
- [`simulador_exemplo1_aula05.html`](./simulador_exemplo1_aula05.html) / [`index.html`](./index.html) — Simulador interativo em HTML5/Canvas (60 FPS) com janelas deslizantes para todos os parâmetros ($U, a, \mu, \rho, \theta, \partial p/\partial x, g, b$) e animação das partículas no meio.
- [`criar_notebook_aula05.py`](./criar_notebook_aula05.py) — Script gerador do notebook e do simulador HTML da Aula 05.
- [`simulacoes_aula04.ipynb`](./simulacoes_aula04.ipynb) & [`animacao_particula_tensores.gif`](./animacao_particula_tensores.gif) — Cinemática e análise tensorial do movimento da partícula fluida (Aula 04).
- [`simulacoes_aula01.ipynb`](./simulacoes_aula01.ipynb) & [`animacao_escoamento_aula01.gif`](./animacao_escoamento_aula01.gif) — Simulações Eulerianas de escoamentos contínuos (Aula 01).
