# SHMC-Net
A Mask-guided Feature Fusion Network for Sperm Head Morphology Classification
-------
Abstract:

Male infertility accounts for about one-third of global infertility cases. Manual assessment of sperm abnormalities through head morphology analysis encounters issues of observer variability and diagnostic discrepancies among experts. Its alternative, Computer-Assisted Semen Analysis (CASA), suffers from low-quality sperm images, small datasets, and noisy class labels. We propose a new approach for sperm head morphology classification, called SHMC-Net, which uses segmentation masks of sperm heads to guide the morphology classification of sperm images. SHMC-Net generates reliable segmentation masks using image priors, refines object boundaries with an efficient graph-based method, and trains an image network with sperm head crops and a mask network with the corresponding masks. In the intermediate stages of the networks, image and mask features are fused with a fusion scheme to better learn morphological features. To handle noisy class labels and regularize training on small datasets, SHMC-Net applies Soft Mixup to combine mixup augmentation and a loss function. We achieve state-of-the-art results on SCIAN and HuSHeM datasets, outperforming methods that use additional pre-training or costly ensembling techniques.

Paper Link: https://arxiv.org/pdf/2402.03697.pdf

** The code for the paper published in ISBI2024 will be published soon.
E-mail the author (nsapkota@nd.edu), if you want access to the codebase for your immediate project. 
