---
layout: archive
title: "CV"
permalink: /cv/
author_profile: true
redirect_from:
  - /resume
---

{% include base_path %}

Research summary
======
Research scientist with over 10 years of experience developing machine learning methods for personalized biology and health. I design and train generative and multimodal models for large-scale single-cell and spatial data, using representation learning, attention, and graph-based modeling to connect cellular states and tissue context with individual disease outcomes. I work hands-on from data harmonization and model implementation through training, evaluation, and biological interpretation. Earlier research applied computer vision and reinforcement learning to personalized health monitoring and digital interventions.

Education
======
* **Postdoc in Statistics**, Harvard University, 2021–2024
  * Supervisor: Professor Susan Murphy
* **M.S./Ph.D. in EECS**, Massachusetts Institute of Technology, 2014–2021
  * Ph.D. Advisors: Professor Thomas Heldt and Professor Vivienne Sze
  * M.S. Advisor: Professor Alan Oppenheim
  * GPA: 5.0/5.0
* **B.S. in EE and Math**, National Taiwan University, 2010–2014
  * Research Advisors: Professor Andy Wu and Professor Benson Yeh
  * GPA: 4.25/4.3 (Rank 1st/198)

Work and research experience
======
* **Scientist II, Brain Health Accelerator**, Allen Institute, June 2024–present
  * Developing AI/ML models for personalized biology and Alzheimer's disease
  * Extending a transformer-based single-cell RNA modeling pipeline toward paired scRNA-seq and scATAC-seq representation learning. Trained RNA models with DeepSpeed across multiple GPUs; now harmonizing brain and immune multiomic datasets and exploring diffusion-based methods to capture complementary regulatory and transcriptional signals.
  * Developed scVIP, a hierarchical generative model with cell-type-aware attention that links single-cell expression and composition to individual phenotypes. Predicted held-out cortical developmental age (Pearson r = 0.95) and Huntington's disease burden (concordance correlation = 0.90), and identified disease-associated cellular populations across neurodegenerative and immune cohorts.
  * Developed spaVI, a generative graph neural network that links shared spatial programs to cell-type-specific expression changes using heterotypic neighbor attention. Integrated Xenium with matched snRNA-seq in Alzheimer's disease caudate, refining cell-state representations and recovering eight spatial dimensions, compared with four using Xenium alone.
  * Co-organized the SEA-AD DREAM Challenge to benchmark computational methods for predicting Alzheimer's disease neuropathology from multi-regional single-cell data.

* **Postdoctoral Fellow, Statistical Reinforcement Learning Lab**, Harvard Statistics, September 2021–May 2024
  * Developing reinforcement learning methods for personalized mobile health interventions
  * Developed reinforcement learning methods for personalized digital health interventions, using graphical models to connect high-frequency behavioral mediators with sparse, delayed health outcomes.
  * Translated Thompson sampling into a behavioral-science study-design framework, mapping actions to intervention options, rewards to proximal outcomes, and priors to interpretable design assumptions.
  * Built a software toolbox and API that enabled behavioral scientists to configure intervention options, outcomes, and priors, then run the resulting Thompson sampling algorithm in mobile health studies.

* **Research Assistant, Energy-Efficient Multimedia Systems Group**, MIT, November 2016–June 2021
  * Using eye movements collected from mobile devices to track neurodegenerative diseases
  * Developed computer vision and signal processing algorithms for mobile-based tracking of neurodegenerative disease, extracting eye-movement and behavioral features from app-based tasks.
  * Built a mobile platform to administer cognitive and behavioral tasks and collect longitudinal eye-movement data for remote health monitoring.
  * Developed Bayesian Gaussian process models to characterize individual-specific trajectories in behavioral features and relate longitudinal changes to disease progression.

* **Research Assistant, Digital Signal Processing Group**, MIT, September 2014–June 2016
  * Amplitude sampling: theory and reconstruction algorithms
  * Analyzed properties of amplitude sampling, a recently proposed approach to representing a signal with quantized amplitudes and continuous time instants.
  * Developed the iterative amplitude sampling reconstruction (IASR) algorithm that empirically converges within much fewer iterations than the most efficient nonuniform sampling reconstruction method (the Voronoi method) under the scheme of amplitude sampling.

* **Summer Intern**, Bose Corporation, June–August 2016
  * Incorporating auditory perception in sound field reconstruction
  * Factored spatial perception into sound field reconstruction by optimizing the time delay and phase delay between the left and right channels.
  * Developed an iterative algorithm that can weigh time delay and phase delay correspondingly for each frequency, which in simulations improved the hearing experience compared with least square optimization.

* **Summer Intern**, Texas Instruments, June–August 2015
  * Developed CNN-based stereo depth estimation to improve the accuracy compared to traditional stereo methods.

* **Undergraduate Research Assistant, Access IC Design Lab**, NTU, September 2012–June 2014
  * Investigated artifact removal of electroencephalography (EEG) using Independent Component Analysis (ICA).
  * Utilized SVM and neural network classifiers for stroke outcome prediction based on arterial blood pressure (ABP).

* **Undergraduate Research Assistant, Communications and Networks Science**, NTU, September 2011–August 2013
  * Investigated distance estimation of concentration-based molecular communication.

Publications
======
  <ul>{% for post in site.publications reversed %}
    {% include archive-single-cv.html %}
  {% endfor %}</ul>

Teaching
======
* **Teaching Assistant, 6.041/6.341 Probabilistic Systems Analysis**, MIT, Spring 2017
  * Held tutorials and office hours to help students with weekly materials and problem sets.
  * Assisted in designing exam questions and grading.
  * Held review sessions to help students encapsulate concepts they learn in class.
* **Teaching Assistant, 6.431 Discrete Time Signal Processing**, MIT, Fall 2016
  * Held recitations and office hours to help students with weekly materials and problem sets.
  * Assisted in designing term projects, exam questions, and grading.
  * Maintained MITx on the edX platform to improve students' learning experience.

Honors
======
* Impact Fellow, Fall 2019
* MIT IBM-AI Best Poster Award, 2019
* MIT MARC (student conference) Presentation and Poster Award, 2018
* Siebel Scholar, 2015
* Jacobs Presidential Fellowship, 2014
* Presidential Awards (6 times out of 8 semesters) at NTU
* InnovateAsia FPGA Workshop and Design Contest Silver Award
* IMO training camp

Service and leadership
======
* SEA-AD DREAM Challenge 2025 organizer
* Speaker at AWIS: Allen Institute
* Mentor of several students during postdoc and grad school
* MIT MARC (student conference) 2018, 2019, 2020, 2021 session chair
* MIT Sidney Pacific (graduate dormitory) trustee, brunch chair, and environment chair
* Vice minister of academic department in student association of EE department at NTU

Skills
======
* **Machine Learning and Statistics**: Bayesian modeling, statistical modeling, Gaussian processes, reinforcement learning, variational autoencoders, representation learning, computer vision, foundation-model-inspired modeling.
* **Biology and Health Data**: single-cell transcriptomics, spatial transcriptomics, multi-omics modeling, longitudinal health data, mobile sensing, Alzheimer's disease modeling.
* **Deep Learning and Infrastructure**: PyTorch, DeepSpeed, transformer and Mamba-based architectures, HPC, Beaker, multi-GPU training.
* **Programming**: Python, R, Matlab, Swift, C++; agentic coding tools (Claude Code).
* **Languages**: English, Chinese.
