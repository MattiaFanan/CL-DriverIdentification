# CL-DriverIdentification
<div id="top"></div>
<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/MattiaFanan/CL-DriverIdentification/tree/main">
    <img src="figures/logo.jpg" alt="Logo" width="150" height="150">
  </a>

  <h1 align="center">Continual Learning for Behavior-based Driver Identification</h1>

  <p align="center">
    Continual Learning for Behavior-based Driver Identification
    <br />
    <a href="https://github.com/MattiaFanan/CL-DriverIdentification/tree/main"><strong>Paper in progress »</strong></a>
    <br />
    <br />
    <a href="https://www.dei.unipd.it/persona/90707c6333c82f24fc7d29bffbe41965">Mattia Fanan</a>
    ·
    <a href="https://www.dei.unipd.it/persona/ce9ba471efd139c3da396eb5662b04c0">Davide Dalle Pezze</a>
    .
    <a href="https://www.dei.unipd.it/persona/1373bd29c9ef0140e39d53ec9add14d2">Emad Efatinasab</a>
    ·
    <a href="https://www.dei.unipd.it/persona/F2BDEDEEDA67FECB0AC87DD91819E093">Ruggero Carli</a>
    .
    <a href="https://www.dei.unipd.it/persona/95DDDDA0C518D43822ADC0338BD38073">Mirco Rampazzo</a>
    .
    <a href="https://www.dei.unipd.it/en/persona/534AC78B8315B31B04D8708B87673B85">Gian Antonio Susto</a>
  </p>
</div>


<div id="abstract"></div>

## 🧩 Abstract

Behavior-based Driver Identification is a significant problem that aims to identify drivers based on their unique driving behaviors. This has numerous applications, including preventing vehicle theft and enhancing personalized driving experiences. While previous studies have shown that Deep Learning approaches can achieve high performance, considerable challenges remain. A key issue is that most studies are conducted offline, rather than directly on vehicles. For real-world applications, models installed in cars must be capable of adapting to new drivers while minimizing resource consumption. Continual Learning techniques, which allow models to learn new driving behaviors while retaining previously acquired knowledge, offer a promising solution to this challenge.

This study addresses the Driver Identification problem in a realistic setting, where the model must adapt to new drivers. We evaluate the performance of various Continual Learning techniques across multiple scenarios with increasing complexity and realistic applicability. Furthermore, given the temporal correlation inherent in this setting, we propose a novel technique called SmooDER. The effectiveness and superiority of our method are validated using the well-known OCSLab dataset.
<p align="right"><a href="#top">(back to top)</a></p>
<div id="usage"></div>

## ⚙️ Usage

To execute the our framework, start by cloning the repository:

```bash
git clone https://github.com/MattiaFanan/CL-DriverIdentification.git

```
<sup>NOTE: if you're accessing this data from the anonymized repository, the above command might not work.</sup>

Then, install the required Python packages by running:

```bash
pip install -r requirements.txt
```

<p align="right"><a href="#top">(back to top)</a></p>
<div id="models"></div>
