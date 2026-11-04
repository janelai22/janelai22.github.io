---
permalink: /
title: "About me"
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---

I am a Scientist II at the [Allen Institute](https://alleninstitute.org/), where I work in the Brain Health Accelerator. I develop machine learning methods for personalized biology and health, with a focus on Alzheimer's disease and other neurodegenerative conditions.

My work centers on generative and multimodal models for large-scale single-cell and spatial data. I am building hierarchical multimodal foundation models for single-cell multiomics, and graph-based models for spatial transcriptomics, that connect cellular states and tissue context to individual disease outcomes. My training in statistics and mathematics shapes how I design models: I aim for methods that are interpretable and that quantify uncertainty. My engineering background means I enjoy working hands-on across the full pipeline, from harmonizing data to building models and pipelines that are scalable and reproducible. Across more than ten years of applying machine learning to health and biology, my goal has been to develop and evaluate models in ways that lead to real biological insight.

Current work
======

- **Multiomic foundation models:** extending a transformer-based single-cell RNA model to paired scRNA-seq and scATAC-seq representation learning, and exploring diffusion-based methods to combine regulatory and transcriptional signals. The work centers on scalability, multimodal fusion design, and downstream evaluation. I am currently improving and evaluating our models on AD brain cells and immune cells.
- **spaVI:** a generative graph neural network that links shared spatial programs, including possibly pathology-induced ones, to cell-type-specific expression changes using heterotypic neighbor attention. Integrating Xenium spatial data with matched snRNA-seq from Alzheimer's disease caudate, it recovers eight spatial dimensions, one of which may relate to Alzheimer's disease mechanisms. I am currently extending it to analyze Abeta and tau pathology in the DFC region.

Earlier work at the Allen Institute
======

- **scVIP:** a hierarchical generative model with cell-type-aware attention that links single-cell expression and cell-type composition to individual phenotypes. It predicts held-out cortical developmental age (Pearson r = 0.95) and Huntington's disease burden (concordance correlation = 0.90), and identifies disease-associated cell populations across neurodegenerative and immune cohorts.
- **SEA-AD DREAM Challenge:** I co-organized this community challenge benchmarking computational methods, including AI agents, for predicting Alzheimer's disease neuropathology from multi-regional single-cell data.

Background
======

Before joining the Allen Institute, I was a postdoctoral fellow in Statistics at Harvard University with Professor Susan Murphy. In the Statistical Reinforcement Learning Lab, I developed reinforcement learning methods for personalized mobile health interventions that incorporate domain scientists' intuitions about causal structure, and built tools that let behavioral scientists design and run Thompson sampling–based studies.

I received my M.S. and Ph.D. in Electrical Engineering and Computer Science from MIT, advised by Professors Thomas Heldt and Vivienne Sze (Ph.D.) and Professor Alan Oppenheim (M.S.). My doctoral research used smartphone cameras and computer vision to measure eye movements, and Bayesian models to track individual neurocognitive state over time, for remote monitoring of neurodegenerative disease. Before that, I earned a B.S. in Electrical Engineering and Mathematics from National Taiwan University.

Research interests
======

- Generative and multimodal models for single-cell and spatial omics, with possible extensions to text and images
- Personalized health, from understanding disease and tracking it over time to designing treatments
- Computer vision, reinforcement learning, Bayesian modeling, and large-scale training
