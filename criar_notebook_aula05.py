import nbformat as nbf
import os
from IPython.display import HTML, display

CODIGOS_DIR = r"G:\Meu Drive\Mestrado - Modelagem Matemática\Mecânica dos fluidos\Códigos"
OUT_NOTEBOOK = os.path.join(CODIGOS_DIR, "simulacoes_aula05.ipynb")
OUT_HTML = os.path.join(CODIGOS_DIR, "simulador_exemplo1_aula05.html")

# Remove o GIF antigo caso exista, mantendo o foco exclusivo no simulador HTML
OLD_GIF = os.path.join(CODIGOS_DIR, "animacao_exemplo1_aula05.gif")
if os.path.exists(OLD_GIF):
    try:
        os.remove(OLD_GIF)
    except Exception:
        pass

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    },
    "language_info": {
        "name": "python",
        "version": "3.13"
    }
}

def md(s):
    nb.cells.append(nbf.v4.new_markdown_cell(s.strip()))

def code(s):
    nb.cells.append(nbf.v4.new_code_cell(s.strip()))

# ==============================================================================
# CÉLULA 0: FORMULAÇÃO MATEMÁTICA CONCISA DO EXEMPLO 1 DA AULA 05
# ==============================================================================
md(r"""# Mecânica dos Fluidos — Aula 05: Soluções Exatas de Navier-Stokes
## Simulador Interativo em HTML5 — Exemplo 1: Escoamento entre Placas Planas Inclinadas
**Universidade Federal de Pelotas (UFPel)**  
**Programa de Pós-Graduação em Modelagem Matemática (PPGMMat)**

---

### Formulação e Expressão de Velocidade Encontrada (Aula 05, p. 2–4)

Considera-se o escoamento laminar, incompressível, permanente e completamente desenvolvido ($u = u(y),\, v = w = 0$) de um fluido newtoniano entre **duas placas planas paralelas** separadas por $h = 2a$ ($y \in [-a, +a]$) e inclinadas de um ângulo $\theta$ com a horizontal:
- **Placa inferior ($y = -a$)**: Fixa $\implies u(-a) = 0$
- **Placa superior ($y = +a$)**: Móvel $\implies u(+a) = U$

Da equação de Navier-Stokes na direção longitudinal $x$:
$$
\mu \frac{\partial^2 u}{\partial y^2} = \rho g \sin\theta + \frac{\partial p}{\partial x} = K
\quad \implies \quad
u(y) = \frac{K}{\mu}\frac{y^2}{2} + C_1 y + C_2
$$
Com as condições de contorno $u(+a) = U$ e $u(-a) = 0$, determinam-se as constantes:
$$
C_1 = \frac{U}{2a}, \qquad C_2 = \frac{U}{2} - \frac{K a^2}{2\mu}
$$
Substituindo $C_1$ e $C_2$, obtém-se a **expressão analítica exata da velocidade**:
$$
\boxed{u(y) = -\frac{K a^2}{2\mu}\left[1 - \frac{y^2}{a^2}\right] + \frac{U}{2}\left[1 + \frac{y}{a}\right]}
\qquad \text{onde} \qquad
\boxed{K = \rho g \sin\theta + \frac{\partial p}{\partial x}}
$$

- **Vazão volumétrica e velocidade média**:
  $$Q = \left(-\frac{2}{3}\frac{K a^2}{\mu} + U\right) a b, \qquad A_T = 2ab, \qquad u_m = -\frac{1}{3}\frac{K a^2}{\mu} + \frac{1}{2}U$$
- **Tensão de cisalhamento**:
  $$\tau(y) = \mu\frac{du}{dy} = K y + \mu\frac{U}{2a}$$""")

# ==============================================================================
# CÓDIGO HTML5/CSS/JS DO SIMULADOR COMPLETO
# ==============================================================================
HTML_SIMULADOR = r"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<style>
  #sim-aula05-root {
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: #f8fafc;
    color: #1e293b;
    border: 1px solid #cbd5e1;
    border-radius: 12px;
    padding: 18px;
    max-width: 1220px;
    margin: 0 auto;
    box-shadow: 0 4px 16px rgba(15, 23, 42, 0.07);
    box-sizing: border-box;
  }
  #sim-aula05-root * {
    box-sizing: border-box;
  }
  .sim-header {
    display: flex;
    flex-wrap: wrap;
    justify-content: space-between;
    align-items: center;
    gap: 10px;
    padding-bottom: 12px;
    margin-bottom: 14px;
    border-bottom: 2px solid #e2e8f0;
  }
  .sim-title {
    margin: 0;
    font-size: 19px;
    font-weight: 700;
    color: #0f172a;
  }
  .sim-subtitle {
    margin: 3px 0 0 0;
    font-size: 13px;
    color: #475569;
  }
  .sim-controls-bar {
    display: flex;
    flex-wrap: wrap;
    gap: 7px;
    align-items: center;
  }
  .sim-btn {
    padding: 6px 12px;
    font-size: 12px;
    font-weight: 600;
    border-radius: 6px;
    border: 1px solid #cbd5e1;
    background: #ffffff;
    color: #1e293b;
    cursor: pointer;
    transition: all 0.15s ease;
  }
  .sim-btn:hover {
    background: #f1f5f9;
    border-color: #94a3b8;
  }
  .sim-btn.primary {
    background: #2563eb;
    color: #ffffff;
    border-color: #1d4ed8;
  }
  .sim-btn.primary:hover {
    background: #1d4ed8;
  }
  .sim-btn.preset-couette {
    background: #eff6ff;
    color: #1d4ed8;
    border-color: #93c5fd;
  }
  .sim-btn.preset-fav {
    background: #f0fdf4;
    color: #15803d;
    border-color: #86efac;
  }
  .sim-btn.preset-adv {
    background: #fef2f2;
    color: #b91c1c;
    border-color: #fca5a5;
  }
  .sliders-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 10px 16px;
    background: #ffffff;
    padding: 14px;
    border-radius: 10px;
    border: 1px solid #e2e8f0;
    margin-bottom: 14px;
  }
  .slider-item {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }
  .slider-label {
    display: flex;
    justify-content: space-between;
    font-size: 12px;
    font-weight: 600;
    color: #334155;
  }
  .slider-val {
    color: #2563eb;
    font-family: 'Consolas', 'Monaco', monospace;
    font-weight: 700;
  }
  .slider-item input[type="range"] {
    width: 100%;
    accent-color: #2563eb;
    cursor: pointer;
    margin: 0;
  }
  .top-row {
    display: grid;
    grid-template-columns: 2.15fr 1fr;
    gap: 14px;
    margin-bottom: 14px;
  }
  .bottom-row {
    display: grid;
    grid-template-columns: 1.15fr 1.15fr 1fr;
    gap: 14px;
  }
  @media (max-width: 960px) {
    .top-row, .bottom-row {
      grid-template-columns: 1fr;
    }
  }
  .panel-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 10px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.03);
    display: flex;
    flex-direction: column;
  }
  .panel-title {
    font-size: 13px;
    font-weight: 700;
    color: #0f172a;
    margin: 0 0 6px 2px;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  .panel-canvas {
    width: 100%;
    height: auto;
    display: block;
    border-radius: 6px;
  }
  .summary-box {
    font-size: 12.5px;
    line-height: 1.55;
    color: #1e293b;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    height: 100%;
  }
  .eq-banner {
    background: #eff6ff;
    border-left: 4px solid #2563eb;
    padding: 8px 10px;
    border-radius: 6px;
    font-family: 'Cambria Math', 'Times New Roman', serif;
    font-size: 14.5px;
    color: #1e3a8a;
    margin-bottom: 8px;
  }
  .metrics-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
  }
  .metrics-table td {
    padding: 4px 6px;
    border-bottom: 1px solid #f1f5f9;
  }
  .metrics-table td:first-child {
    color: #475569;
    font-weight: 600;
  }
  .metrics-table td:last-child {
    text-align: right;
    font-family: 'Consolas', monospace;
    font-weight: 700;
    color: #0f172a;
  }
  .badge-status {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 999px;
    font-size: 11px;
    font-weight: 700;
  }
  .badge-ok { background: #dcfce7; color: #166534; }
  .badge-rev { background: #fee2e2; color: #991b1b; }
</style>

<div id="sim-aula05-root">
  <!-- CABEÇALHO E BOTÕES DE CONTROLE -->
  <div class="sim-header">
    <div>
      <h2 class="sim-title">Simulador Interativo em HTML5 — Exemplo 1 da Aula 05 (Navier-Stokes)</h2>
      <p class="sim-subtitle">
        Escoamento Laminar de Couette-Poiseuille entre Duas Placas Planas Inclinadas &nbsp;|&nbsp;
        <b>u(y) = -(Ka²/2μ)[1 - (y/a)²] + (U/2)[1 + y/a]</b>
      </p>
    </div>
    <div class="sim-controls-bar">
      <button id="btn-play" class="sim-btn primary">⏸ Pausar</button>
      <button id="btn-reset" class="sim-btn">↺ Reiniciar Linha Material</button>
      <button id="btn-p-couette" class="sim-btn preset-couette">Couette Puro (K = 0)</button>
      <button id="btn-p-fav" class="sim-btn preset-fav">Favorável (K &lt; 0)</button>
      <button id="btn-p-adv" class="sim-btn preset-adv">Adverso / Refluxo (K &gt; 0)</button>
    </div>
  </div>

  <!-- JANELAS DESLIZANTES (SLIDERS DE PARÂMETROS) -->
  <div class="sliders-grid">
    <div class="slider-item">
      <div class="slider-label">
        <span>Velocidade da Placa Móvel U (m/s)</span>
        <span id="lbl-U" class="slider-val">1.50</span>
      </div>
      <input id="inp-U" type="range" min="-3.0" max="5.0" step="0.05" value="1.50">
    </div>

    <div class="slider-item">
      <div class="slider-label">
        <span>Meia-altura do Canal a = h/2 (m)</span>
        <span id="lbl-a" class="slider-val">0.050</span>
      </div>
      <input id="inp-a" type="range" min="0.015" max="0.150" step="0.005" value="0.050">
    </div>

    <div class="slider-item">
      <div class="slider-label">
        <span>Viscosidade Dinâmica μ (Pa·s)</span>
        <span id="lbl-mu" class="slider-val">0.020</span>
      </div>
      <input id="inp-mu" type="range" min="0.002" max="0.100" step="0.001" value="0.020">
    </div>

    <div class="slider-item">
      <div class="slider-label">
        <span>Massa Específica ρ (kg/m³)</span>
        <span id="lbl-rho" class="slider-val">1000</span>
      </div>
      <input id="inp-rho" type="range" min="500" max="1500" step="25" value="1000">
    </div>

    <div class="slider-item">
      <div class="slider-label">
        <span>Inclinação das Placas θ (graus)</span>
        <span id="lbl-theta" class="slider-val">10.0°</span>
      </div>
      <input id="inp-theta" type="range" min="-45" max="45" step="1" value="10">
    </div>

    <div class="slider-item">
      <div class="slider-label">
        <span>Gradiente de Pressão ∂p/∂x (Pa/m)</span>
        <span id="lbl-dpdx" class="slider-val">-2000</span>
      </div>
      <input id="inp-dpdx" type="range" min="-5000" max="4000" step="50" value="-2000">
    </div>

    <div class="slider-item">
      <div class="slider-label">
        <span>Gravidade g (m/s²)</span>
        <span id="lbl-g" class="slider-val">9.81</span>
      </div>
      <input id="inp-g" type="range" min="0.0" max="20.0" step="0.05" value="9.81">
    </div>

    <div class="slider-item">
      <div class="slider-label">
        <span>Largura Transversal b (m)</span>
        <span id="lbl-b" class="slider-val">1.00</span>
      </div>
      <input id="inp-b" type="range" min="0.10" max="5.00" step="0.10" value="1.00">
    </div>

    <div class="slider-item">
      <div class="slider-label">
        <span>Nº de Partículas no Meio</span>
        <span id="lbl-npart" class="slider-val">200</span>
      </div>
      <input id="inp-npart" type="range" min="40" max="450" step="10" value="200">
    </div>

    <div class="slider-item">
      <div class="slider-label">
        <span>Velocidade de Reprodução</span>
        <span id="lbl-speed" class="slider-val">1.0x</span>
      </div>
      <input id="inp-speed" type="range" min="0.1" max="3.0" step="0.1" value="1.0">
    </div>
  </div>

  <!-- LINHA SUPERIOR: ANIMAÇÃO DO CANAL + GEOMETRIA INCLINADA -->
  <div class="top-row">
    <div class="panel-card">
      <div class="panel-title">
        <span id="title-canal">Animação das Partículas no Meio Fluido</span>
        <span id="telemetry-pill" style="font-size: 11.5px; color: #1d4ed8; background: #eff6ff; padding: 2px 8px; border-radius: 5px; font-family: monospace;"></span>
      </div>
      <canvas id="cv-main" width="760" height="350" class="panel-canvas"></canvas>
    </div>

    <div class="panel-card">
      <div class="panel-title">
        <span id="title-geo">Geometria Inclinada (θ) e Gravidade</span>
      </div>
      <canvas id="cv-geo" width="360" height="350" class="panel-canvas"></canvas>
    </div>
  </div>

  <!-- LINHA INFERIOR: DECOMPOSIÇÃO DE u(y) + TENSÃO CISALHANTE tau(y) + RESUMO MATEMÁTICO -->
  <div class="bottom-row">
    <div class="panel-card">
      <div class="panel-title">
        <span>Decomposição da Expressão de Velocidade u(y)</span>
      </div>
      <canvas id="cv-vel" width="400" height="295" class="panel-canvas"></canvas>
    </div>

    <div class="panel-card">
      <div class="panel-title">
        <span id="title-tau">Perfil de Tensão Cisalhante τ(y)</span>
      </div>
      <canvas id="cv-tau" width="400" height="295" class="panel-canvas"></canvas>
    </div>

    <div class="panel-card">
      <div class="panel-title">
        <span>Parâmetros Calculados e Solução Exata</span>
        <span id="badge-refluxo" class="badge-status badge-ok">Sem Refluxo</span>
      </div>
      <div class="summary-box">
        <div class="eq-banner">
          <b>u(y)</b> = − [K·a² / (2μ)]·(1 − y²/a²) + (U / 2)·(1 + y/a)<br>
          <b>τ(y)</b> = K·y + μ·U / (2a) &nbsp;|&nbsp; <b>K</b> = ρ·g·sen(θ) + ∂p/∂x
        </div>
        <table class="metrics-table">
          <tr><td>Fator combinado K:</td><td id="tb-K">—</td></tr>
          <tr><td>Constante C₁ = U/(2a):</td><td id="tb-C1">—</td></tr>
          <tr><td>Constante C₂ = u(0):</td><td id="tb-C2">—</td></tr>
          <tr><td>Velocidade Placa Sup. u(+a):</td><td id="tb-ua">—</td></tr>
          <tr><td>Velocidade Placa Inf. u(-a):</td><td id="tb-uma">0.000 m/s</td></tr>
          <tr><td>Faixa [u_min, u_max]:</td><td id="tb-uminmax">—</td></tr>
          <tr><td>Velocidade Média u_m:</td><td id="tb-um">—</td></tr>
          <tr><td>Vazão Volumétrica Q:</td><td id="tb-Q">—</td></tr>
          <tr><td>Tensões [τ(-a), τ(+a)]:</td><td id="tb-tau">—</td></tr>
        </table>
      </div>
    </div>
  </div>
</div>

<script>
(function() {
  const inpU     = document.getElementById('inp-U');
  const inpA     = document.getElementById('inp-a');
  const inpMu    = document.getElementById('inp-mu');
  const inpRho   = document.getElementById('inp-rho');
  const inpTheta = document.getElementById('inp-theta');
  const inpDpdx  = document.getElementById('inp-dpdx');
  const inpG     = document.getElementById('inp-g');
  const inpB     = document.getElementById('inp-b');
  const inpNpart = document.getElementById('inp-npart');
  const inpSpeed = document.getElementById('inp-speed');

  const btnPlay  = document.getElementById('btn-play');
  const btnReset = document.getElementById('btn-reset');

  const cvMain = document.getElementById('cv-main');
  const ctxMain = cvMain.getContext('2d');
  const cvGeo  = document.getElementById('cv-geo');
  const ctxGeo = cvGeo.getContext('2d');
  const cvVel  = document.getElementById('cv-vel');
  const ctxVel = cvVel.getContext('2d');
  const cvTau  = document.getElementById('cv-tau');
  const ctxTau = cvTau.getContext('2d');

  let isPlaying = true;
  let simTime = 0.0;
  let topHatchOffset = 0.0;

  const MAX_PARTICLES = 450;
  const particles = [];
  for (let i = 0; i < MAX_PARTICLES; i++) {
    particles.push({
      x: Math.random(),                   // Posição longitudinal normalizada [0, 1]
      eta: -0.95 + 1.90 * Math.random(),  // Posição transversal adimensional y/a in [-0.95, 0.95]
      r: 3.2 + 1.2 * Math.random()
    });
  }

  const N_LINE = 55;
  let matLine = [];
  function resetMaterialLine() {
    simTime = 0.0;
    matLine = [];
    for (let i = 0; i < N_LINE; i++) {
      const eta = -1.0 + (2.0 * i) / (N_LINE - 1);
      matLine.push({ x: 0.16, eta: eta });
    }
  }
  resetMaterialLine();

  btnPlay.addEventListener('click', () => {
    isPlaying = !isPlaying;
    btnPlay.textContent = isPlaying ? '⏸ Pausar' : '▶ Reproduzir';
  });

  btnReset.addEventListener('click', resetMaterialLine);

  document.getElementById('btn-p-couette').addEventListener('click', () => {
    inpTheta.value = 0; inpDpdx.value = 0; inpU.value = 1.6; inpMu.value = 0.020; inpA.value = 0.050;
    resetMaterialLine();
  });
  document.getElementById('btn-p-fav').addEventListener('click', () => {
    inpTheta.value = 5; inpDpdx.value = -2500; inpU.value = 1.5; inpMu.value = 0.020; inpA.value = 0.050;
    resetMaterialLine();
  });
  document.getElementById('btn-p-adv').addEventListener('click', () => {
    inpTheta.value = 15; inpDpdx.value = 1200; inpU.value = 1.5; inpMu.value = 0.020; inpA.value = 0.050;
    resetMaterialLine();
  });

  function readState() {
    const U        = parseFloat(inpU.value);
    const a        = parseFloat(inpA.value);
    const mu       = parseFloat(inpMu.value);
    const rho      = parseFloat(inpRho.value);
    const thetaDeg = parseFloat(inpTheta.value);
    const dpdx     = parseFloat(inpDpdx.value);
    const g        = parseFloat(inpG.value);
    const b        = parseFloat(inpB.value);
    const nPart    = parseInt(inpNpart.value, 10);
    const speed    = parseFloat(inpSpeed.value);

    const thetaRad = thetaDeg * Math.PI / 180.0;
    const K = rho * g * Math.sin(thetaRad) + dpdx;
    const C1 = U / (2.0 * a);
    const C2 = 0.5 * U - (K * a * a) / (2.0 * mu);
    const um = -(1.0 / 3.0) * (K * a * a) / mu + 0.5 * U;
    const AT = 2.0 * a * b;
    const Q = um * AT;
    const tauBot = -K * a + mu * C1;
    const tauTop =  K * a + mu * C1;

    return { U, a, mu, rho, thetaDeg, thetaRad, dpdx, g, b, nPart, speed, K, C1, C2, um, AT, Q, tauBot, tauTop };
  }

  function evalProfile(eta, s) {
    const uPois = -(s.K * s.a * s.a) / (2.0 * s.mu) * (1.0 - eta * eta);
    const uCou  = 0.5 * s.U * (1.0 + eta);
    const u     = uPois + uCou;
    const tau   = s.K * (eta * s.a) + s.mu * s.C1;
    return { u, uPois, uCou, tau };
  }

  // Mapa de cores científico Coolwarm (Azul -> Branco/Neutro -> Vermelho)
  function coolwarmRGB(t) {
    // t in [0, 1]
    t = Math.max(0, Math.min(1, t));
    let r, g, b;
    if (t < 0.5) {
      const f = t * 2.0;
      r = Math.round(59 + f * (247 - 59));
      g = Math.round(76 + f * (247 - 76));
      b = Math.round(192 + f * (247 - 192));
    } else {
      const f = (t - 0.5) * 2.0;
      r = Math.round(247 + f * (180 - 247));
      g = Math.round(247 - f * (247 - 4));
      b = Math.round(247 - f * (247 - 38));
    }
    return `rgb(${r},${g},${b})`;
  }

  function updateDOM(s, uMin, uMax) {
    document.getElementById('lbl-U').textContent     = s.U.toFixed(2);
    document.getElementById('lbl-a').textContent     = s.a.toFixed(3);
    document.getElementById('lbl-mu').textContent    = s.mu.toFixed(3);
    document.getElementById('lbl-rho').textContent   = s.rho.toFixed(0);
    document.getElementById('lbl-theta').textContent = s.thetaDeg.toFixed(1) + '°';
    document.getElementById('lbl-dpdx').textContent  = s.dpdx.toFixed(0);
    document.getElementById('lbl-g').textContent     = s.g.toFixed(2);
    document.getElementById('lbl-b').textContent     = s.b.toFixed(2);
    document.getElementById('lbl-npart').textContent = s.nPart;
    document.getElementById('lbl-speed').textContent = s.speed.toFixed(1) + 'x';

    document.getElementById('title-canal').textContent =
      `Animação das Partículas no Meio Fluido (θ = ${s.thetaDeg.toFixed(1)}°, K = ${s.K.toFixed(1)} N/m³)`;
    document.getElementById('telemetry-pill').textContent =
      `t = ${simTime.toFixed(2)} s | u_m = ${s.um.toFixed(3)} m/s | Q = ${(s.Q * 1000).toFixed(2)} L/s`;
    document.getElementById('title-tau').textContent =
      `Perfil de Tensão Cisalhante: τ(-a)=${s.tauBot.toFixed(1)} Pa | τ(+a)=${s.tauTop.toFixed(1)} Pa`;

    document.getElementById('tb-K').textContent       = `${s.K.toFixed(2)} N/m³`;
    document.getElementById('tb-C1').textContent      = `${s.C1.toFixed(3)} s⁻¹`;
    document.getElementById('tb-C2').textContent      = `${s.C2.toFixed(4)} m/s`;
    document.getElementById('tb-ua').textContent      = `${s.U.toFixed(3)} m/s`;
    document.getElementById('tb-uminmax').textContent = `[${uMin.toFixed(3)}, ${uMax.toFixed(3)}] m/s`;
    document.getElementById('tb-um').textContent      = `${s.um.toFixed(4)} m/s`;
    document.getElementById('tb-Q').textContent       = `${s.Q.toFixed(5)} m³/s (${(s.Q*1000).toFixed(2)} L/s)`;
    document.getElementById('tb-tau').textContent     = `[${s.tauBot.toFixed(2)}, ${s.tauTop.toFixed(2)}] Pa`;

    const badge = document.getElementById('badge-refluxo');
    if (uMin < -1e-3) {
      badge.textContent = 'Escoamento Reverso (u < 0)';
      badge.className = 'badge-status badge-rev';
    } else {
      badge.textContent = 'Escoamento Unidirecional';
      badge.className = 'badge-status badge-ok';
    }
  }

  function drawMainChannel(s, uMin, uMax, uAbsMax) {
    const W = cvMain.width, H = cvMain.height;
    ctxMain.clearRect(0, 0, W, H);

    // Área do canal com espaço reservado à direita para a barra de cores (colorbar) e abaixo para legenda
    const x0 = 58, x1 = W - 92;
    const yTop = 30, yBot = H - 72;
    const chW = x1 - x0;
    const chH = yBot - yTop;
    const yMid = 0.5 * (yTop + yBot);

    const vMinCmap = uMin < 0 ? -uAbsMax : Math.min(0, uMin);
    const vMaxCmap = Math.max(uAbsMax, 0.01);

    // 1. Pintura do campo de velocidades u(y) ao fundo
    const nStrips = 70;
    const stripH = chH / nStrips;
    for (let i = 0; i < nStrips; i++) {
      const py = yTop + i * stripH;
      const eta = (yMid - (py + 0.5 * stripH)) / (0.5 * chH);
      const uVal = evalProfile(eta, s).u;
      const norm = (uVal - vMinCmap) / (vMaxCmap - vMinCmap + 1e-9);
      ctxMain.fillStyle = coolwarmRGB(norm);
      ctxMain.globalAlpha = 0.34;
      ctxMain.fillRect(x0, py, chW, stripH + 0.8);
    }
    ctxMain.globalAlpha = 1.0;

    // 2. Grade sutil e linha de centro y = 0
    ctxMain.strokeStyle = '#cbd5e1';
    ctxMain.lineWidth = 1;
    ctxMain.setLineDash([3, 3]);
    for (let gx = 0.2; gx < 1.0; gx += 0.2) {
      const px = x0 + gx * chW;
      ctxMain.beginPath(); ctxMain.moveTo(px, yTop); ctxMain.lineTo(px, yBot); ctxMain.stroke();
    }
    ctxMain.strokeStyle = '#64748b';
    ctxMain.setLineDash([6, 5]);
    ctxMain.beginPath(); ctxMain.moveTo(x0, yMid); ctxMain.lineTo(x1, yMid); ctxMain.stroke();
    ctxMain.setLineDash([]);

    // 3. Envelope fixo de referência do perfil de velocidades em x = 0.16 * L
    const xRef = x0 + 0.16 * chW;
    ctxMain.strokeStyle = '#1e3a8a';
    ctxMain.lineWidth = 1.2;
    ctxMain.setLineDash([2, 3]);
    ctxMain.beginPath(); ctxMain.moveTo(xRef, yTop); ctxMain.lineTo(xRef, yBot); ctxMain.stroke();

    ctxMain.strokeStyle = '#2563eb';
    ctxMain.lineWidth = 2.0;
    ctxMain.setLineDash([5, 4]);
    ctxMain.beginPath();
    for (let i = 0; i <= 50; i++) {
      const eta = -1.0 + (2.0 * i) / 50;
      const py = yMid - eta * (0.5 * chH);
      const uVal = evalProfile(eta, s).u;
      const px = xRef + (uVal / uAbsMax) * (0.20 * chW);
      if (i === 0) ctxMain.moveTo(px, py); else ctxMain.lineTo(px, py);
    }
    ctxMain.stroke();
    ctxMain.setLineDash([]);

    // 4. Atualização e desenho das partículas no meio fluido
    const dtVis = isPlaying ? (0.0024 * s.speed) : 0.0;
    if (isPlaying) {
      simTime += dtVis * (1.0 / uAbsMax) * 10.0;
    }

    for (let i = 0; i < s.nPart; i++) {
      const pt = particles[i];
      const uVal = evalProfile(pt.eta, s).u;
      pt.x = (pt.x + (uVal / uAbsMax) * dtVis) % 1.0;
      if (pt.x < 0) pt.x += 1.0;

      const px = x0 + pt.x * chW;
      const py = yMid - pt.eta * (0.5 * chH);
      const norm = (uVal - vMinCmap) / (vMaxCmap - vMinCmap + 1e-9);

      ctxMain.fillStyle = coolwarmRGB(norm);
      ctxMain.strokeStyle = '#1e293b';
      ctxMain.lineWidth = 0.8;
      ctxMain.beginPath();
      ctxMain.arc(px, py, pt.r, 0, 2 * Math.PI);
      ctxMain.fill();
      ctxMain.stroke();
    }

    // 5. Linha Material Fluida x(y,t) (amarela com pontos pretos/dourados)
    for (let i = 0; i < N_LINE; i++) {
      const lp = matLine[i];
      const uVal = evalProfile(lp.eta, s).u;
      lp.x = (lp.x + (uVal / uAbsMax) * dtVis) % 1.0;
      if (lp.x < 0) lp.x += 1.0;
    }

    ctxMain.strokeStyle = '#eab308';
    ctxMain.lineWidth = 2.6;
    ctxMain.beginPath();
    for (let i = 0; i < N_LINE; i++) {
      const lp = matLine[i];
      const px = x0 + lp.x * chW;
      const py = yMid - lp.eta * (0.5 * chH);
      if (i > 0 && Math.abs(lp.x - matLine[i - 1].x) < 0.35) {
        ctxMain.lineTo(px, py);
      } else {
        ctxMain.moveTo(px, py);
      }
    }
    ctxMain.stroke();

    for (let i = 0; i < N_LINE; i += 2) {
      const lp = matLine[i];
      const px = x0 + lp.x * chW;
      const py = yMid - lp.eta * (0.5 * chH);
      ctxMain.fillStyle = '#facc15';
      ctxMain.strokeStyle = '#0f172a';
      ctxMain.lineWidth = 1.0;
      ctxMain.beginPath();
      ctxMain.arc(px, py, 3.2, 0, 2 * Math.PI);
      ctxMain.fill();
      ctxMain.stroke();
    }

    // 6. Placa Superior Móvel (y = +a, vermelha) e Placa Inferior Fixa (y = -a, preta)
    ctxMain.fillStyle = '#dc2626';
    ctxMain.fillRect(x0, yTop - 6, chW, 6);
    topHatchOffset = (topHatchOffset + (s.U / uAbsMax) * dtVis * chW) % 20;
    ctxMain.strokeStyle = '#fecaca';
    ctxMain.lineWidth = 1.4;
    for (let hx = x0 - 20; hx < x1 + 20; hx += 20) {
      const tx = hx + topHatchOffset;
      if (tx >= x0 && tx <= x1 - 5) {
        ctxMain.beginPath(); ctxMain.moveTo(tx, yTop - 6); ctxMain.lineTo(tx + 5, yTop); ctxMain.stroke();
      }
    }

    ctxMain.fillStyle = '#0f172a';
    ctxMain.fillRect(x0, yBot, chW, 6);

    // 7. Eixos e Rótulos do Canal
    ctxMain.fillStyle = '#334155';
    ctxMain.font = '11px Segoe UI, sans-serif';
    ctxMain.textAlign = 'right';
    ctxMain.fillText(`+${s.a.toFixed(3)}`, x0 - 6, yTop + 4);
    ctxMain.fillText('0.000', x0 - 6, yMid + 4);
    ctxMain.fillText(`-${s.a.toFixed(3)}`, x0 - 6, yBot + 4);

    ctxMain.save();
    ctxMain.translate(14, yMid);
    ctxMain.rotate(-Math.PI / 2);
    ctxMain.textAlign = 'center';
    ctxMain.font = '600 11.5px Segoe UI, sans-serif';
    ctxMain.fillText('Posição Transversal y [m]', 0, 0);
    ctxMain.restore();

    ctxMain.textAlign = 'center';
    for (let gx = 0.0; gx <= 1.001; gx += 0.2) {
      const px = x0 + gx * chW;
      ctxMain.fillText(gx.toFixed(1), px, yBot + 20);
    }
    ctxMain.font = '600 11.5px Segoe UI, sans-serif';
    ctxMain.fillText('Posição Longitudinal ao longo das placas x [m]', x0 + 0.5 * chW, yBot + 38);

    // 8. Barra de Cores (Colorbar) à direita
    const cbX = x1 + 16, cbW = 14;
    for (let i = 0; i < chH; i++) {
      const norm = 1.0 - (i / chH);
      ctxMain.fillStyle = coolwarmRGB(norm);
      ctxMain.fillRect(cbX, yTop + i, cbW, 1.5);
    }
    ctxMain.strokeStyle = '#94a3b8';
    ctxMain.strokeRect(cbX, yTop, cbW, chH);

    ctxMain.fillStyle = '#334155';
    ctxMain.font = '10.5px Segoe UI, sans-serif';
    ctxMain.textAlign = 'left';
    ctxMain.fillText(vMaxCmap.toFixed(2), cbX + cbW + 5, yTop + 8);
    ctxMain.fillText((0.5 * (vMaxCmap + vMinCmap)).toFixed(2), cbX + cbW + 5, yMid + 4);
    ctxMain.fillText(vMinCmap.toFixed(2), cbX + cbW + 5, yBot);

    ctxMain.save();
    ctxMain.translate(W - 8, yMid);
    ctxMain.rotate(-Math.PI / 2);
    ctxMain.textAlign = 'center';
    ctxMain.font = '600 11px Segoe UI, sans-serif';
    ctxMain.fillText('Velocidade u(y) [m/s]', 0, 0);
    ctxMain.restore();

    // 9. Legenda horizontal limpa no rodapé (sem sobrepor o gráfico)
    const legY = H - 10;
    ctxMain.font = '11px Segoe UI, sans-serif';
    ctxMain.textAlign = 'left';

    ctxMain.fillStyle = '#dc2626';
    ctxMain.fillRect(x0, legY - 8, 18, 4);
    ctxMain.fillStyle = '#334155';
    ctxMain.fillText(`Placa Sup. Móvel (u = ${s.U.toFixed(2)} m/s)`, x0 + 23, legY - 3);

    ctxMain.fillStyle = '#0f172a';
    ctxMain.fillRect(x0 + 205, legY - 8, 18, 4);
    ctxMain.fillStyle = '#334155';
    ctxMain.fillText('Placa Inf. Fixa (u = 0)', x0 + 228, legY - 3);

    ctxMain.strokeStyle = '#2563eb';
    ctxMain.lineWidth = 2;
    ctxMain.setLineDash([4, 3]);
    ctxMain.beginPath(); ctxMain.moveTo(x0 + 365, legY - 6); ctxMain.lineTo(x0 + 385, legY - 6); ctxMain.stroke();
    ctxMain.setLineDash([]);
    ctxMain.fillText('Envelope u(y)', x0 + 390, legY - 3);

    ctxMain.strokeStyle = '#eab308';
    ctxMain.lineWidth = 2.5;
    ctxMain.beginPath(); ctxMain.moveTo(x0 + 485, legY - 6); ctxMain.lineTo(x0 + 505, legY - 6); ctxMain.stroke();
    ctxMain.fillStyle = '#facc15';
    ctxMain.beginPath(); ctxMain.arc(x0 + 495, legY - 6, 3, 0, 2 * Math.PI); ctxMain.fill();
    ctxMain.fillStyle = '#334155';
    ctxMain.fillText('Linha Material x(y,t)', x0 + 510, legY - 3);
  }

  function drawGeometry(s) {
    const W = cvGeo.width, H = cvGeo.height;
    ctxGeo.clearRect(0, 0, W, H);

    const cx = W * 0.5, cy = H * 0.48;
    const L = 210, hVis = 54;

    // Grade de fundo
    ctxGeo.strokeStyle = '#f1f5f9';
    ctxGeo.lineWidth = 1;
    for (let x = 20; x < W; x += 35) {
      ctxGeo.beginPath(); ctxGeo.moveTo(x, 10); ctxGeo.lineTo(x, H - 10); ctxGeo.stroke();
    }
    for (let y = 20; y < H; y += 35) {
      ctxGeo.beginPath(); ctxGeo.moveTo(10, y); ctxGeo.lineTo(W - 10, y); ctxGeo.stroke();
    }

    // Linha horizontal de referência
    ctxGeo.strokeStyle = '#64748b';
    ctxGeo.setLineDash([5, 4]);
    ctxGeo.beginPath();
    ctxGeo.moveTo(cx - 115, cy + 35);
    ctxGeo.lineTo(cx + 115, cy + 35);
    ctxGeo.stroke();
    ctxGeo.setLineDash([]);

    // Canal rotacionado por theta
    ctxGeo.save();
    ctxGeo.translate(cx, cy);
    ctxGeo.rotate(-s.thetaRad);

    ctxGeo.fillStyle = 'rgba(125, 211, 252, 0.40)';
    ctxGeo.fillRect(-L / 2, -hVis / 2, L, hVis);

    // Placa superior móvel
    ctxGeo.strokeStyle = '#dc2626';
    ctxGeo.lineWidth = 4.5;
    ctxGeo.beginPath(); ctxGeo.moveTo(-L / 2, -hVis / 2); ctxGeo.lineTo(L / 2, -hVis / 2); ctxGeo.stroke();

    // Seta de velocidade U na placa superior
    ctxGeo.strokeStyle = '#dc2626';
    ctxGeo.lineWidth = 2;
    const dirU = s.U >= 0 ? 1 : -1;
    ctxGeo.beginPath();
    ctxGeo.moveTo(20, -hVis / 2 - 12);
    ctxGeo.lineTo(20 + dirU * 45, -hVis / 2 - 12);
    ctxGeo.stroke();
    ctxGeo.fillStyle = '#dc2626';
    ctxGeo.font = 'bold 11px Segoe UI, sans-serif';
    ctxGeo.fillText(`U = ${s.U.toFixed(2)} m/s`, 15, -hVis / 2 - 18);

    // Placa inferior fixa
    ctxGeo.strokeStyle = '#0f172a';
    ctxGeo.lineWidth = 4.5;
    ctxGeo.beginPath(); ctxGeo.moveTo(-L / 2, hVis / 2); ctxGeo.lineTo(L / 2, hVis / 2); ctxGeo.stroke();

    // Eixos locais x e y
    ctxGeo.strokeStyle = '#334155';
    ctxGeo.lineWidth = 1.5;
    ctxGeo.beginPath();
    ctxGeo.moveTo(-L / 2 + 18, 0); ctxGeo.lineTo(-L / 2 + 52, 0);
    ctxGeo.moveTo(-L / 2 + 18, 0); ctxGeo.lineTo(-L / 2 + 18, -28);
    ctxGeo.stroke();
    ctxGeo.fillStyle = '#0f172a';
    ctxGeo.font = 'bold 11px Segoe UI, sans-serif';
    ctxGeo.fillText('x', -L / 2 + 56, 4);
    ctxGeo.fillText('y', -L / 2 + 14, -32);

    ctxGeo.restore();

    // Vetor gravidade vertical g
    ctxGeo.strokeStyle = '#15803d';
    ctxGeo.fillStyle = '#15803d';
    ctxGeo.lineWidth = 2.8;
    const gy0 = cy - 15, gy1 = cy + 75;
    ctxGeo.beginPath(); ctxGeo.moveTo(cx, gy0); ctxGeo.lineTo(cx, gy1); ctxGeo.stroke();
    ctxGeo.beginPath();
    ctxGeo.moveTo(cx, gy1 + 4);
    ctxGeo.lineTo(cx - 5, gy1 - 6);
    ctxGeo.lineTo(cx + 5, gy1 - 6);
    ctxGeo.closePath();
    ctxGeo.fill();

    ctxGeo.font = 'bold 13px Segoe UI, sans-serif';
    ctxGeo.fillText('g', cx + 10, gy1 - 12);

    // Rótulos de componentes da gravidade e inclinação
    const gxVal = -s.g * Math.sin(s.thetaRad);
    const gyVal = -s.g * Math.cos(s.thetaRad);
    ctxGeo.fillStyle = '#1e293b';
    ctxGeo.font = '12px Segoe UI, sans-serif';
    ctxGeo.textAlign = 'center';
    ctxGeo.fillText(`θ = ${s.thetaDeg.toFixed(1)}°  |  h = 2a = ${(2 * s.a).toFixed(3)} m`, cx, 26);
    ctxGeo.fillStyle = '#15803d';
    ctxGeo.fillText(`gₓ = -g·sen(θ) = ${gxVal.toFixed(2)} m/s²`, cx, H - 38);
    ctxGeo.fillText(`gᵧ = -g·cos(θ) = ${gyVal.toFixed(2)} m/s²`, cx, H - 18);
    ctxGeo.textAlign = 'left';
  }

  function drawVelocityDecomposition(s) {
    const W = cvVel.width, H = cvVel.height;
    ctxVel.clearRect(0, 0, W, H);

    const padL = 54, padR = 20, padT = 18, padB = 56;
    const pw = W - padL - padR;
    const ph = H - padT - padB;
    const yMid = padT + 0.5 * ph;

    // Determina escala horizontal englobando as 3 curvas (Couette, Poiseuille e Total)
    let minV = 0, maxV = 0;
    for (let i = 0; i <= 60; i++) {
      const eta = -1.0 + (2.0 * i) / 60;
      const p = evalProfile(eta, s);
      minV = Math.min(minV, p.u, p.uCou, p.uPois, s.um);
      maxV = Math.max(maxV, p.u, p.uCou, p.uPois, s.um);
    }
    const span = Math.max(maxV - minV, 0.2);
    const xMin = minV - 0.1 * span;
    const xMax = maxV + 0.1 * span;

    const toX = (v) => padL + ((v - xMin) / (xMax - xMin)) * pw;
    const toY = (eta) => yMid - eta * (0.5 * ph);

    // Grade
    ctxVel.strokeStyle = '#e2e8f0';
    ctxVel.lineWidth = 1;
    ctxVel.strokeRect(padL, padT, pw, ph);
    ctxVel.beginPath();
    ctxVel.moveTo(padL, yMid); ctxVel.lineTo(padL + pw, yMid);
    ctxVel.stroke();

    // Linha vertical u = 0
    if (xMin <= 0 && xMax >= 0) {
      const xZero = toX(0);
      ctxVel.strokeStyle = '#64748b';
      ctxVel.setLineDash([4, 4]);
      ctxVel.beginPath(); ctxVel.moveTo(xZero, padT); ctxVel.lineTo(xZero, padT + ph); ctxVel.stroke();
      ctxVel.setLineDash([]);
    }

    // 1. Parcela Couette (vermelha tracejada)
    ctxVel.strokeStyle = '#dc2626';
    ctxVel.lineWidth = 1.8;
    ctxVel.setLineDash([5, 4]);
    ctxVel.beginPath();
    for (let i = 0; i <= 50; i++) {
      const eta = -1.0 + (2.0 * i) / 50;
      const px = toX(evalProfile(eta, s).uCou);
      const py = toY(eta);
      if (i === 0) ctxVel.moveTo(px, py); else ctxVel.lineTo(px, py);
    }
    ctxVel.stroke();

    // 2. Parcela Pressão + Gravidade (verde traço-ponto)
    ctxVel.strokeStyle = '#16a34a';
    ctxVel.lineWidth = 1.8;
    ctxVel.beginPath();
    for (let i = 0; i <= 50; i++) {
      const eta = -1.0 + (2.0 * i) / 50;
      const px = toX(evalProfile(eta, s).uPois);
      const py = toY(eta);
      if (i === 0) ctxVel.moveTo(px, py); else ctxVel.lineTo(px, py);
    }
    ctxVel.stroke();

    // 3. Linha de Velocidade Média u_m (roxa tracejada)
    const xUm = toX(s.um);
    ctxVel.strokeStyle = '#9333ea';
    ctxVel.lineWidth = 1.5;
    ctxVel.beginPath(); ctxVel.moveTo(xUm, padT); ctxVel.lineTo(xUm, padT + ph); ctxVel.stroke();
    ctxVel.setLineDash([]);

    // 4. Velocidade Resultante u(y) (azul espessa)
    ctxVel.strokeStyle = '#2563eb';
    ctxVel.lineWidth = 3.0;
    ctxVel.beginPath();
    for (let i = 0; i <= 60; i++) {
      const eta = -1.0 + (2.0 * i) / 60;
      const px = toX(evalProfile(eta, s).u);
      const py = toY(eta);
      if (i === 0) ctxVel.moveTo(px, py); else ctxVel.lineTo(px, py);
    }
    ctxVel.stroke();

    // Rótulos dos eixos
    ctxVel.fillStyle = '#334155';
    ctxVel.font = '10.5px Segoe UI, sans-serif';
    ctxVel.textAlign = 'right';
    ctxVel.fillText(`+${s.a.toFixed(2)}`, padL - 5, padT + 8);
    ctxVel.fillText('0.00', padL - 5, yMid + 4);
    ctxVel.fillText(`-${s.a.toFixed(2)}`, padL - 5, padT + ph);

    ctxVel.textAlign = 'center';
    for (let k = 0; k <= 4; k++) {
      const vVal = xMin + (k / 4) * (xMax - xMin);
      const px = toX(vVal);
      ctxVel.fillText(vVal.toFixed(1), px, padT + ph + 15);
    }
    ctxVel.font = '600 11px Segoe UI, sans-serif';
    ctxVel.fillText('Velocidade u(y) [m/s]', padL + 0.5 * pw, padT + ph + 31);

    // Legenda compacta na base
    ctxVel.font = '10.5px Segoe UI, sans-serif';
    ctxVel.textAlign = 'left';
    const ly = H - 7;
    ctxVel.fillStyle = '#2563eb'; ctxVel.fillText('▬ u(y) Total', 12, ly);
    ctxVel.fillStyle = '#dc2626'; ctxVel.fillText('--- Couette', 95, ly);
    ctxVel.fillStyle = '#16a34a'; ctxVel.fillText('--- Pressão/Grav.', 175, ly);
    ctxVel.fillStyle = '#9333ea'; ctxVel.fillText(`--- u_m (${s.um.toFixed(2)})`, 285, ly);
  }

  function drawShearStress(s) {
    const W = cvTau.width, H = cvTau.height;
    ctxTau.clearRect(0, 0, W, H);

    const padL = 54, padR = 20, padT = 18, padB = 56;
    const pw = W - padL - padR;
    const ph = H - padT - padB;
    const yMid = padT + 0.5 * ph;

    const minT = Math.min(0, s.tauBot, s.tauTop);
    const maxT = Math.max(0, s.tauBot, s.tauTop);
    const span = Math.max(maxT - minT, 0.2);
    const xMin = minT - 0.15 * span;
    const xMax = maxT + 0.15 * span;

    const toX = (t) => padL + ((t - xMin) / (xMax - xMin)) * pw;
    const toY = (eta) => yMid - eta * (0.5 * ph);

    ctxTau.strokeStyle = '#e2e8f0';
    ctxTau.lineWidth = 1;
    ctxTau.strokeRect(padL, padT, pw, ph);
    ctxTau.beginPath(); ctxTau.moveTo(padL, yMid); ctxTau.lineTo(padL + pw, yMid); ctxTau.stroke();

    // Linha vertical tau = 0
    if (xMin <= 0 && xMax >= 0) {
      const xZero = toX(0);
      ctxTau.strokeStyle = '#64748b';
      ctxTau.setLineDash([4, 4]);
      ctxTau.beginPath(); ctxTau.moveTo(xZero, padT); ctxTau.lineTo(xZero, padT + ph); ctxTau.stroke();
      ctxTau.setLineDash([]);
    }

    // Preenchimento sombreado de tau(y)
    const xZ = toX(0);
    ctxTau.fillStyle = 'rgba(249, 115, 22, 0.15)';
    ctxTau.beginPath();
    ctxTau.moveTo(xZ, toY(-1));
    ctxTau.lineTo(toX(s.tauBot), toY(-1));
    ctxTau.lineTo(toX(s.tauTop), toY(1));
    ctxTau.lineTo(xZ, toY(1));
    ctxTau.closePath();
    ctxTau.fill();

    // Reta tau(y) = K y + mu U / (2a)
    ctxTau.strokeStyle = '#ea580c';
    ctxTau.lineWidth = 3.0;
    ctxTau.beginPath();
    ctxTau.moveTo(toX(s.tauBot), toY(-1));
    ctxTau.lineTo(toX(s.tauTop), toY(1));
    ctxTau.stroke();

    // Pontos nas paredes y = -a e y = +a
    ctxTau.fillStyle = '#dc2626';
    ctxTau.beginPath(); ctxTau.arc(toX(s.tauBot), toY(-1), 4.5, 0, 2 * Math.PI); ctxTau.fill();
    ctxTau.beginPath(); ctxTau.arc(toX(s.tauTop), toY(1), 4.5, 0, 2 * Math.PI); ctxTau.fill();

    // Rótulos
    ctxTau.fillStyle = '#334155';
    ctxTau.font = '10.5px Segoe UI, sans-serif';
    ctxTau.textAlign = 'right';
    ctxTau.fillText(`+${s.a.toFixed(2)}`, padL - 5, padT + 8);
    ctxTau.fillText('0.00', padL - 5, yMid + 4);
    ctxTau.fillText(`-${s.a.toFixed(2)}`, padL - 5, padT + ph);

    ctxTau.textAlign = 'center';
    for (let k = 0; k <= 4; k++) {
      const tVal = xMin + (k / 4) * (xMax - xMin);
      const px = toX(tVal);
      ctxTau.fillText(tVal.toFixed(1), px, padT + ph + 15);
    }
    ctxTau.font = '600 11px Segoe UI, sans-serif';
    ctxTau.fillText('Tensão Cisalhante τ(y) = K·y + μ·U/(2a) [Pa]', padL + 0.5 * pw, padT + ph + 31);

    ctxTau.font = '10.5px Segoe UI, sans-serif';
    ctxTau.fillStyle = '#ea580c';
    ctxTau.fillText(`▬ τ(y) linear  |  τ(-a) = ${s.tauBot.toFixed(2)} Pa  |  τ(+a) = ${s.tauTop.toFixed(2)} Pa`, padL + 0.5 * pw, H - 7);
  }

  function renderLoop() {
    const s = readState();

    // Calcula extremos de velocidade u(y) em [-a, a]
    let uMin = Infinity, uMax = -Infinity;
    for (let i = 0; i <= 80; i++) {
      const eta = -1.0 + (2.0 * i) / 80;
      const uVal = evalProfile(eta, s).u;
      if (uVal < uMin) uMin = uVal;
      if (uVal > uMax) uMax = uVal;
    }
    const uAbsMax = Math.max(Math.abs(uMin), Math.abs(uMax), Math.abs(s.U), 0.1);

    updateDOM(s, uMin, uMax);
    drawMainChannel(s, uMin, uMax, uAbsMax);
    drawGeometry(s);
    drawVelocityDecomposition(s);
    drawShearStress(s);

    requestAnimationFrame(renderLoop);
  }

  requestAnimationFrame(renderLoop);
})();
</script>
</html>
"""

# ==============================================================================
# CÉLULA 1: EXIBIÇÃO DO SIMULADOR HTML NO NOTEBOOK E GRAVAÇÃO DO ARQUIVO .HTML
# ==============================================================================
code(f'''from IPython.display import HTML, display

html_simulador_aula05 = r\"\"\"{HTML_SIMULADOR}\"\"\"

# Salva também como arquivo .html independente na pasta Códigos
with open("simulador_exemplo1_aula05.html", "w", encoding="utf-8") as f:
    f.write(html_simulador_aula05)

display(HTML(html_simulador_aula05))''')

# Preenche o output da célula 1 diretamente no .ipynb para abrir já renderizado
nb.cells[1].execution_count = 1
nb.cells[1].outputs = [
    nbf.v4.new_output(
        output_type="display_data",
        data={
            "text/html": HTML_SIMULADOR,
            "text/plain": "<IPython.core.display.HTML object>"
        },
        metadata={}
    )
]

with open(OUT_HTML, "w", encoding="utf-8") as f_html:
    f_html.write(HTML_SIMULADOR)

nbf.write(nb, OUT_NOTEBOOK)
print(f"[OK] Notebook salvo em: {OUT_NOTEBOOK}")
print(f"[OK] Simulador HTML independente salvo em: {OUT_HTML}")
