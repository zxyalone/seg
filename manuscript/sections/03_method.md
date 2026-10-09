# 3 Method

## 3.1 Overview

Fig. 1 summarizes PhASER-3D [[TBD: expansion of the acronym "PhASER"]]. The backbone is the 3D nnU-Net `PlainConvUNet` \cite{isensee2021nnunet}. It has six encoder stages with 32, 64, 128, 256, 320 and 320 channels, takes $128^3$ patches and uses deep supervision. Encoder and decoder are initialized from VoComni weights \cite{wu2024voco,wu2024largescale}. Of 292 parameter tensors, 282 are transferred. The 10 segmentation-head tensors are re-initialized because the output changes from 21 to 3 classes (background, liver, tumor).

Let $X_l \in \mathbb{R}^{C_l \times D_l \times H_l \times W_l}$, $l = 1,\dots,5$, denote the five non-bottleneck encoder skips. For a $128^3$ patch, $D_l = H_l = W_l = 128/2^{\,l-1}$ and $C_l \in \{32, 64, 128, 256, 320\}$. PhASER-3D adds two components to the backbone. A tumor-prior head (Section 3.4) reads $X_1$ and $X_2$ and outputs a voxel-wise tumor probability $P$. One Phase–Frequency Coherence Routing (PFCR) module per skip (Section 3.3) then rewrites each skip:

$$
\tilde{X}_l = \mathrm{PFCR}_l\!\left(X_l,\, P\right), \qquad l = 1,\dots,5. \tag{1}
$$

The decoder is unchanged and receives $\tilde{X}_l$ in place of $X_l$; the bottleneck is untouched. Inside each PFCR, a phase congruency (PC) map of the skip (Section 3.2) and the prior $P$ predict a per-voxel distribution over three Haar frequency bands. This distribution reweights the skip's wavelet sub-bands. The five PFCR modules do not share parameters. Each one is an exact identity mapping at initialization. Training therefore starts from the function computed by the pretrained backbone with its re-initialized head.

## 3.2 Monogenic phase congruency with learnable log-Gabor scales

**Single-channel proxy.** PC is computed on a learned scalar summary of the skip rather than on all $C_l$ channels:

$$
p_l = \mathbf{w}_l^{\top} X_l, \qquad \mathbf{w}_l \in \mathbb{R}^{C_l}, \quad \mathbf{w}_l^{(0)} = \tfrac{1}{C_l}\mathbf{1}, \tag{2}
$$

that is, a bias-free $1\times1\times1$ convolution initialized to the channel mean. Every spectral operation below acts on $p_l$, so its cost does not depend on $C_l$. We drop the index $l$ in the rest of this subsection.

**Learnable log-Gabor bank.** Let $\hat{p}(\mathbf{u}) = \mathcal{F}\{p\}(\mathbf{u})$, where $\mathbf{u} = (u_z, u_y, u_x)$ is frequency in cycles per voxel and $\rho = \lVert \mathbf{u} \rVert_2$. We use $S = 3$ isotropic log-Gabor filters \cite{kovesi1999phase}:

$$
G_s(\rho) = \exp\!\left( -\frac{\ln^2(\rho/\rho_s)}{2\,\ln^2 \sigma_s} \right), \qquad G_s(0) = 0, \qquad s = 1,\dots,S. \tag{3}
$$

Here $\rho_s$ is the center frequency and $\sigma_s$ is the bandwidth ratio ($\kappa/\omega_0$ in Kovesi's notation). Both are learned per scale and per skip through the reparameterization

$$
\rho_s = \tfrac{1}{2}\,\mathrm{sigmoid}(\theta^{\rho}_s), \qquad \sigma_s = 0.05 + 0.9\,\mathrm{sigmoid}(\theta^{\sigma}_s). \tag{4}
$$

Eq. (4) keeps $\rho_s$ below the Nyquist frequency of $1/2$ and keeps $\sigma_s$ in $(0.05, 0.95)$, so that $\ln \sigma_s \neq 0$. Center wavelengths are initialized at $\lambda_s = \lambda_{\min}\, m^{s-1}$ with $\lambda_{\min} = 3$ voxels and $m = 2.1$. This gives $\lambda = (3.0, 6.3, 13.2)$ voxels, i.e. $\rho = (0.333, 0.159, 0.076)$. The bandwidth ratio is initialized at $\sigma_s = 0.55$, which Kovesi associates with an approximately two-octave bandwidth \cite{kovesi1999phase}. Because $G_s(0) = 0$, the bank has no DC response.

**Monogenic responses.** The Riesz transform generalizes the Hilbert transform to $n$ dimensions. With a radial band-pass filter, it yields an isotropic quadrature triple in 2D \cite{felsberg2001monogenic}, and a quadruple in 3D, without an orientation filter bank. Its transfer function along axis $j$ is $i\,u_j/\rho$; the sign convention is immaterial here because only norms enter the PC. The even and odd responses at scale $s$ are

$$
e_s = \mathcal{F}^{-1}\!\left\{ G_s\, \hat{p} \right\}, \qquad
o_{s,j} = \mathcal{F}^{-1}\!\left\{ i\,\frac{u_j}{\rho}\, G_s\, \hat{p} \right\}, \quad j \in \{z, y, x\}. \tag{5}
$$

We stack them into the monogenic vector $\mathbf{v}_s = (e_s, o_{s,z}, o_{s,y}, o_{s,x}) \in \mathbb{R}^4$, with local amplitude

$$
A_s = \lVert \mathbf{v}_s \rVert_2 . \tag{6}
$$

Local energy and the amplitude sum are

$$
E = \Big\lVert \sum_{s=1}^{S} \mathbf{v}_s \Big\rVert_2, \qquad \Sigma_A = \sum_{s=1}^{S} A_s . \tag{7}
$$

By the triangle inequality, $E \le \Sigma_A$. Equality holds only where all $\mathbf{v}_s$ are parallel, that is, where local phase and orientation agree across scales.

**Noise threshold.** We follow Kovesi's noise model \cite{kovesi1999phase}. Filter responses to additive white noise are assumed to be Rayleigh distributed. The finest-scale filter has the widest bandwidth and therefore the strongest noise response. The median of a Rayleigh variable with scale $\tau$ is $\tau\sqrt{2\ln 2} = \tau\sqrt{\ln 4}$. We therefore estimate, per sample and over all voxels of the patch,

$$
\tau = \frac{\operatorname{median}_{\mathbf{x}}\, \mathrm{sg}\!\left[A_1(\mathbf{x})\right]}{\sqrt{\ln 4}}, \tag{8}
$$

where $\mathrm{sg}[\cdot]$ stops the gradient and $s = 1$ is the scale initialized as the finest. The noise contributions of all scales are accumulated with the gain

$$
g = \sum_{s=1}^{S} \frac{\rho_s}{\max_{s'} \rho_{s'}} . \tag{9}
$$

At initialization, $g = \sum_{s} m^{-(s-1)} = (1 - m^{-S})/(1 - m^{-1}) = 1.703$. This is the geometric factor of Kovesi's estimator. Once the learned $\rho_s$ are no longer geometrically spaced, Eq. (9) uses their actual ratios. The threshold is the mean plus $k$ standard deviations of a Rayleigh distribution with scale $\tau g$. This mirrors Kovesi's noise-circle radius $T = \mu_R + k\,\sigma_R$:

$$
T = \tau\, g \left( \sqrt{\pi/2} + k\, \sqrt{(4-\pi)/2} \right), \qquad k = \mathrm{softplus}(\theta^{k}), \tag{10}
$$

with $k$ initialized at 2. The median is detached for two reasons. It is a noise-level estimate, not a feature. Its gradient would also flow through a single, arbitrarily selected voxel.

**Phase congruency.** The PC map is

$$
\mathrm{PC} = \operatorname{clip}_{[0,1]}\!\left( \frac{\left[\, E - T \,\right]_+}{\Sigma_A + \epsilon\, \overline{\Sigma_A} + \epsilon_0} \right), \tag{11}
$$

where $[\cdot]_+ = \max(\cdot, 0)$ and $\overline{\Sigma_A}$ is the spatial mean of $\Sigma_A$ over the patch. We use $\epsilon = 10^{-3}$, and $\epsilon_0 = 10^{-12}$ only guards against division by zero. Without $T$, Eq. (11) reduces to the ratio $E/(\Sigma_A + \epsilon)$ used by FSIM \cite{zhang2011fsim}. FSIM computes that ratio with oriented 2D log-Gabor pairs; here it is computed from the isotropic 3D monogenic signal. [[V2-pending: hard threshold $[E-T]_+$ vs. a soft threshold]]

**Contrast invariance.** For any $a > 0$ and $b \in \mathbb{R}$,

$$
\mathrm{PC}\!\left[a\,p + b\right] = \mathrm{PC}\!\left[p\right] \tag{12}
$$

holds exactly, up to $\epsilon_0$. The argument has three steps. First, $G_s(0) = 0$ removes the offset $b$. Second, Eqs. (5)–(7) are linear in $p$ followed by norms, so $e_s$, $o_{s,j}$, $A_s$, $E$ and $\Sigma_A$ are positively homogeneous of degree 1. Third, the median in Eq. (8) scales by $a$, so $T$ is also of degree 1, and so is the relative floor $\epsilon\,\overline{\Sigma_A}$. Eq. (11) is then a ratio of degree-1 quantities, which has degree 0. An absolute $\epsilon$, as in \cite{kovesi1999phase,zhang2011fsim}, would break Eq. (12) when amplitudes are small. That is the low-contrast regime this module targets.

Two limits of Eq. (12) matter. (i) The invariance concerns the operator and its input $p_l$. It does not make the network invariant to CT contrast, because $X_l$ is a nonlinear function of the image. (ii) $T$ is estimated from the same input. PC is therefore invariant to global rescaling but not to the local signal-to-noise ratio. A margin whose energy falls inside the noise circle yields $\mathrm{PC} = 0$ whatever its scale. Section 4.8 and Fig. 3 examine this behavior.

**Implementation.** Under automatic mixed precision (AMP), the whole spectral block runs in fp32 with autocast disabled, because half-precision FFTs have limited dynamic range and limited operator support. We use real-input FFTs. One forward transform of $p$ is followed by a single batched inverse transform over the $S(1+3) = 12$ complex kernels of Eq. (5). Frequency grids are in cycles per voxel and are cached per input shape. To bound cost, a proxy with more than $2^{20}$ voxels is downsampled by powers of two (trilinear, at least 8 voxels per axis). PC is computed at the reduced size and upsampled back. With $128^3$ patches, this affects only $l = 1$, whose PC is computed at $64^3$. [[V2-pending: keep or remove the $2^{20}$-voxel cap]]

## 3.3 Phase–Frequency Coherence Routing (PFCR)

Fig. 2 details one PFCR module. It has three steps: a routing network, a Haar band decomposition, and a gated residual.

**Routing network.** The prior $P$ is resampled trilinearly to the skip resolution, giving $P_l$. A small network maps PC and prior to a distribution over three bands:

$$
R_l = \operatorname{softmax}\!\Big( \mathbf{W}_2 * \delta\big( \mathbf{W}_1 * [\,\mathrm{PC}_l \,;\, P_l\,] + \mathbf{b}_1 \big) + \mathbf{b}_2 \Big) \in \Delta^{2} \ \text{per voxel}. \tag{13}
$$

$\mathbf{W}_1$ is a $3\times3\times3$ convolution from 2 to 8 channels, $\delta$ is ReLU, and $\mathbf{W}_2$ is a $1\times1\times1$ convolution from 8 to 3 channels. The softmax runs over channels, so $R_l = (r^{\mathrm{low}}, r^{\mathrm{mid}}, r^{\mathrm{high}})$ with $\sum_g r^g = 1$. $\mathbf{W}_2$ and $\mathbf{b}_2$ are zero-initialized, so $R_l \equiv (\tfrac13, \tfrac13, \tfrac13)$ at the start of training. The router sees only $\mathrm{PC}_l$ and $P_l$, not the features. Each routing decision is therefore an explicit function of phase evidence and lesion likelihood, which can be inspected directly (Fig. 5).

**Haar sub-bands.** We apply a single-level, separable 3D Haar transform. Along each axis, analysis and synthesis are

$$
a_L[n] = \tfrac{1}{2}\big(x[2n] + x[2n{+}1]\big), \quad a_H[n] = \tfrac{1}{2}\big(x[2n] - x[2n{+}1]\big); \qquad x[2n] = a_L[n] + a_H[n], \quad x[2n{+}1] = a_L[n] - a_H[n]. \tag{14}
$$

This pair reconstructs perfectly: $\mathrm{IDWT} \circ \mathrm{DWT} = \mathrm{id}$. Applying it along $z$, $y$ and $x$ yields eight sub-bands $X_l^{(b)} \in \mathbb{R}^{C_l \times \frac{D_l}{2} \times \frac{H_l}{2} \times \frac{W_l}{2}}$, indexed by $b = (b_z, b_y, b_x) \in \{L, H\}^3$. Let $h(b)$ count the high-pass axes of $b$. The three groups are

$$
\mathcal{G}_{\mathrm{low}} = \{ b : h(b) = 0 \} = \{\mathrm{LLL}\}, \quad
\mathcal{G}_{\mathrm{mid}} = \{ b : h(b) = 1 \}, \quad
\mathcal{G}_{\mathrm{high}} = \{ b : h(b) \ge 2 \}. \tag{15}
$$

$\mathcal{G}_{\mathrm{mid}}$ holds the three sub-bands that are high-pass along exactly one axis. $\mathcal{G}_{\mathrm{high}}$ holds the four with at least two high-pass axes (three with two, plus HHH). The low group carries smooth content such as parenchyma. The mid group carries variation along a single axis. The high group carries variation along several axes at once, as at curved margins, small blobs and fine texture.

**Band mixing and residual.** $R_l$ is downsampled trilinearly to the sub-band grid, giving $\bar{R}_l$, and broadcast over channels. With $g(b)$ the group of sub-band $b$,

$$
Y_l = \mathrm{IDWT}\!\left( \left\{ \bar{r}_l^{\,g(b)} \odot X_l^{(b)} \right\}_{b \in \{L,H\}^3} \right), \tag{16}
$$

$$
\tilde{X}_l = X_l + \tanh(\gamma_l)\, Y_l, \qquad \gamma_l^{(0)} = 0. \tag{17}
$$

Odd spatial sizes are zero-padded before the DWT and cropped afterwards; this never occurs with $128^3$ patches.

**Interpretation.** Let $\Pi_g$ be the operator that keeps the sub-bands of group $g$ and inverts, so that $\sum_g \Pi_g = \mathrm{id}$. Where $R_l$ is locally constant, the linearity of the IDWT turns Eqs. (16)–(17) into

$$
\tilde{X}_l = \sum_{g \in \{\mathrm{low}, \mathrm{mid}, \mathrm{high}\}} \big( 1 + \tanh(\gamma_l)\, r_l^{g} \big)\, \Pi_g X_l . \tag{18}
$$

PFCR is thus a spatially varying, three-band equalizer. Each band gain lies in $(0, 2)$. The total extra gain, $\tanh(\gamma_l)$, is set by the residual gate, and the route only redistributes it among bands. At initialization, $Y_l = X_l/3$ and $\tanh(0) = 0$, so $\tilde{X}_l = X_l$ exactly. Zero-gated residuals are a common way to add modules to a pretrained network. Here the identity start protects the VoComni features while the new parameters are still random.

**Gradient flow at initialization.** The identity start has a side effect. At $\gamma_l = 0$, $\partial \tilde{X}_l / \partial R_l = 0$ and $\partial \tilde{X}_l / \partial \mathrm{PC}_l = 0$. The segmentation loss therefore reaches only $\gamma_l$, through $\partial \tilde{X}_l/\partial \gamma_l = Y_l = X_l/3$. The router, the proxy and the log-Gabor parameters receive segmentation gradients only after $\gamma_l$ moves away from zero, and those gradients are scaled by $\tanh(\gamma_l)$. The auxiliary losses $\mathcal{L}_{\mathrm{rank}}$ and $\mathcal{L}_{\mathrm{budget}}$ (Section 3.5) act directly on $\mathrm{PC}_l$ and $R_l$. They give these parameters a gradient path that bypasses $\tanh(\gamma_l)$. The zero-initialized $\mathbf{W}_2$ does not block this path, because $\partial R_l / \partial \mathbf{W}_2$ is proportional to the hidden activations, which are non-zero.

## 3.4 Tumor-prior head

The prior head (called the small-tumor head in our code) runs at the highest resolution, where small lesions still span several voxels. It reads the raw skips $X_1$ ($32 \times 128^3$) and $X_2$ ($64 \times 64^3$) before any PFCR:

$$
F = \big[\, \phi_1(X_1) \,;\, \mathcal{U}(\phi_2(X_2)) \,\big] \in \mathbb{R}^{64 \times 128^3}, \tag{19}
$$

$$
Z = \psi_2\Big( \delta\big( \mathrm{IN}( \psi_1(F) ) \big) \Big) \in \mathbb{R}^{2 \times 128^3}, \tag{20}
$$

$$
P = \operatorname{softmax}(Z)_{\mathrm{tumor}} = \mathrm{sigmoid}(Z_{\mathrm{tumor}} - Z_{\mathrm{bg}}). \tag{21}
$$

$\phi_1$ ($32 \to 32$) and $\phi_2$ ($64 \to 32$) are $3\times3\times3$ convolutions. $\mathcal{U}$ is trilinear $2\times$ upsampling. $\psi_1$ is a $3\times3\times3$ convolution ($64 \to 32$), IN is affine instance normalization, and $\psi_2$ is a $1\times1\times1$ convolution ($32 \to 2$). The head is supervised by $\mathcal{L}_{\mathrm{prior}}$ from the first epoch (Section 3.5). It also runs at inference, because PFCR consumes $P$. $P$ is not detached before the router, so the router also back-propagates into the head.

**Parameter and computational overhead.** The following counts follow exactly from the module definitions.

- Each PFCR module has $C_l + 475$ parameters. These are $C_l$ proxy weights, 7 log-Gabor and threshold parameters ($3\,\theta^\rho$, $3\,\theta^\sigma$, $\theta^k$), 467 router parameters and $\gamma_l$. <!-- router: W1 = 2*8*27 + 8 = 440; W2 = 8*3*1 + 3 = 27; total 467. 467 + 7 + 1 = 475. -->
- The five PFCR modules add $\sum_l C_l + 5 \times 475 = 800 + 2{,}375 = 3{,}175$ parameters. <!-- sum C_l = 32+64+128+256+320 = 800 -->
- The prior head adds $138{,}466$ parameters. <!-- phi1 = 32*32*27+32 = 27,680; phi2 = 64*32*27+32 = 55,328; psi1 = 64*32*27+32 = 55,328; IN affine = 2*32 = 64; psi2 = 32*2+2 = 66; total 138,466 -->
- In total, PhASER-3D adds $141{,}641$ parameters, or [[TBD: % of the backbone's parameter count]].

On the spectral side, each skip and sample needs one forward real FFT plus 12 inverse transforms on $N'_l$ voxels. Here $N'_l = 64^3, 64^3, 32^3, 16^3, 8^3$, where the first value reflects the voxel cap. This totals $13 \times 561{,}664 \approx 7.30\times10^6$ transformed voxels per sample, at $O(N'_l \log N'_l)$ each and independent of $C_l$. <!-- 2*262,144 + 32,768 + 4,096 + 512 = 561,664 --> Dense convolutions add, per $128^3$ patch, about 1.09 GMAC for the routers and 0.09 GMAC for the proxies. The prior head adds 188.6 GMAC, because $\phi_1$ and $\psi_1$ run at full resolution. <!-- routers: (2*8*27 + 8*3) = 456 MAC/voxel x sum_l N_l (2,396,672) = 1.093e9; proxies: sum_l C_l N_l = 8.93e7; head: 27,648*128^3 + 55,296*64^3 + 55,296*128^3 + 64*128^3 = 1.886e11 --> Backbone MACs, wall-clock time and memory are reported in Table 7 [[TBD: measured]].

## 3.5 Training objective

**Segmentation loss.** We keep nnU-Net's deep-supervised Dice + cross-entropy loss \cite{isensee2021nnunet}:

$$
\mathcal{L}_{\mathrm{seg}} = \sum_{k=0}^{K-1} \omega_k \Big[ \mathcal{L}_{\mathrm{CE}}(\hat{Y}_k, Y_k) + \mathcal{L}_{\mathrm{DC}}(\hat{Y}_k, Y_k) \Big], \qquad
\omega_k = \frac{2^{-k}}{\sum_{k'=0}^{K-2} 2^{-k'}} \ (k < K-1), \quad \omega_{K-1} = 0. \tag{22}
$$

$\hat{Y}_k$ is the decoder output at scale $k$. $Y_k$ is the label map downsampled by nearest neighbor. $\mathcal{L}_{\mathrm{DC}}$ is the soft Dice loss over the foreground classes. With $K = 5$ outputs, $\omega = (0.533, 0.267, 0.133, 0.067, 0)$.

**PC boundary-ranking loss.** Let $M_l$ be the tumor mask, downsampled by nearest neighbor to the resolution of skip $l$. Let $\oplus_k$ and $\ominus_k$ denote binary dilation and erosion with a $k\times k\times k$ cube. We define three regions:

$$
B_l = (M_l \oplus_3) \setminus (M_l \ominus_3), \qquad
I_l = M_l \ominus_3, \qquad
O_l = (M_l \oplus_7) \setminus (M_l \cup B_l). \tag{23}
$$

$B_l$ is a two-voxel shell across the tumor boundary. $I_l$ is the eroded interior. $O_l$ is the peritumoral ring 2–3 voxels outside the lesion (Chebyshev distance). The positives are $B_l$, and the negatives are $N_l = I_l \cup O_l$. For sample $i$, let $\mu^{+}_{l,i}$ and $\mu^{-}_{l,i}$ be the mean of $\mathrm{PC}_{l,i}$ over $B_l$ and over $N_l$. Then

$$
\mathcal{L}_{\mathrm{rank}} = \frac{1}{|\mathcal{L}'|} \sum_{l \in \mathcal{L}'} \frac{1}{|\mathcal{V}_l|} \sum_{i \in \mathcal{V}_l} \Big[\, \Delta - \big( \mu^{+}_{l,i} - \mu^{-}_{l,i} \big) \Big]_+, \qquad \Delta = 0.2. \tag{24}
$$

$\mathcal{V}_l$ is the set of samples in which both $B_l$ and $N_l$ are non-empty, and $\mathcal{L}'$ is the set of skips with $\mathcal{V}_l \neq \emptyset$. The loss is zero for batches without tumor.

We use ranking rather than regressing PC onto a boundary mask. PC is a generic structure detector. It responds correctly to vessels, the liver capsule and other true edges. A dense regression target would penalize those responses and change what the operator measures. Eq. (24) constrains only the contrast between the lesion margin and its immediate surroundings. The hinge becomes inactive once that contrast reaches $\Delta$, after which the loss no longer pulls on the filters. The negatives do not include vessels; Section 5 discusses this limitation.

**Frequency-budget loss.** For skip $l$ and sample $i$, let $\bar{R}^{\mathrm{in}}_{l,i}$ and $\bar{R}^{\mathrm{out}}_{l,i}$ be the mean route vectors inside $M_l$ and over all other voxels of the patch. The second set includes liver, other tissue and air. Then

$$
\mathcal{L}_{\mathrm{budget}} = \frac{1}{5B} \sum_{l=1}^{5} \sum_{i=1}^{B} \frac{1}{3} \Big( \mathbb{1}[M_{l,i} \neq \emptyset]\, \big\lVert \bar{R}^{\mathrm{in}}_{l,i} - \boldsymbol{\beta}^{\mathrm{in}} \big\rVert_1 + \big\lVert \bar{R}^{\mathrm{out}}_{l,i} - \boldsymbol{\beta}^{\mathrm{out}} \big\rVert_1 \Big), \tag{25}
$$

with $\boldsymbol{\beta}^{\mathrm{in}} = (0.20, 0.40, 0.40)$ and $\boldsymbol{\beta}^{\mathrm{out}} = (0.70, 0.22, 0.08)$ in (low, mid, high) order. The budgets encode a prior: lesions should keep relatively more mid- and high-band content than the background, which is dominated by smooth parenchyma and air. Eq. (25) constrains only region means, so the router remains free to vary routes within each region. [[TBD: state how the budget values were chosen]]

**Prior loss.** The prior head is trained with a class-weighted cross-entropy at $128^3$:

$$
\mathcal{L}_{\mathrm{prior}} = \frac{1}{|\Omega|} \sum_{v \in \Omega} w_v\, \mathrm{CE}\big(Z_v, M_v\big), \qquad w_v = 1 + 3\, M_v. \tag{26}
$$

Tumor voxels therefore have weight 4. $\Omega$ is the set of all voxels in the batch, and the sum is normalized by $|\Omega|$, not by $\sum_v w_v$.

**Total loss.** With epoch index $e = 0, 1, \dots$,

$$
\mathcal{L} = \mathcal{L}_{\mathrm{seg}} + \psi(e) \Big( 0.2\, \mathcal{L}_{\mathrm{rank}} + 0.05\, \mathcal{L}_{\mathrm{budget}} + 0.1\, \mathcal{L}_{\mathrm{prior}} \Big), \tag{27}
$$

$$
\psi(e) = \min\!\left(1, \frac{e+1}{5}\right). \tag{28}
$$

All auxiliary terms are active from the first epoch and reach full weight in the fifth. Supervising the prior from the start matters because PFCR conditions on $P$ from the start. [[TBD: rationale for the weights 0.2 / 0.05 / 0.1, or state that they were fixed a priori]]

**Optimization.** We use SGD with Nesterov momentum 0.99 and base learning rate $\eta_0 = 5\times10^{-3}$. The learning rate is scaled per parameter group: $0.3\,\eta_0$ for the pretrained encoder, $0.5\,\eta_0$ for the decoder, and $\eta_0$ for the PhASER parameters (PFCR and prior head). A polynomial schedule is applied once per epoch with a shared factor:

$$
\eta_g(e) = \eta_g(0)\, \big(1 - e/E\big)^{0.9}, \qquad E = 200, \tag{29}
$$

Eq. (29) preserves the group ratios. By contrast, nnU-Net's default poly scheduler overwrites all groups with a single rate. Weight decay $3\times10^{-5}$ applies to all convolution weights and biases. It does not apply to the scalar PhASER parameters $\theta^\rho_s$, $\theta^\sigma_s$, $\theta^k$ and $\gamma_l$. Decay would pull these toward 0, which corresponds to $\rho_s = 0.25$, $\sigma_s = 0.5$, $k = \ln 2$ and a closed gate. Those targets are arbitrary and unrelated to the initialization. Gradients are clipped at norm 12, and training uses AMP, with the PC block in fp32 (Section 3.2). Epochs, iterations, batch size and sampling are listed in Section 4.2 and Table 2.

## Figure specifications

**Fig. 1: Overall architecture (full page width).**

- *Layout.* A U-shaped diagram, read left to right. Draw the encoder column on the left and the decoder column on the right.
- *Input.* A CT patch box labelled "CT patch $1\times128^3$".
- *Encoder.* Six stacked boxes, E1 to E6, with channel and resolution: E1 $32\times128^3$, E2 $64\times64^3$, E3 $128\times32^3$, E4 $256\times16^3$, E5 $320\times8^3$, E6 (bottleneck) $320\times4^3$. Fill them blue and label them "VoComni-pretrained".
- *Decoder.* Mirror boxes D5 to D1 at the same resolutions. Fill them light blue.
- *Deep-supervision heads.* Small outputs from D1 to D5 with shapes $3\times128^3, \dots, 3\times8^3$ and weights $\omega_k$ (0.533 / 0.267 / 0.133 / 0.067 / 0). Mark the final output "segmentation".
- *Skip connections.* Draw each horizontal skip from E$l$ to D$l$ ($l = 1..5$) passing through an orange box "PFCR$_l$". Write the tensor shape on the arrow both before and after the box; they are identical.
- *Bottleneck.* E6 connects to D5 directly, with no PFCR.
- *Prior head.* A green box "Tumor-prior head" below E1 and E2. It takes arrows from E1 and E2, tapped before PFCR. Its output is "$P$, $1\times128^3$". Draw dashed green arrows from $P$ to every PFCR$_l$, labelled "trilinear resample".
- *Identity note.* A small note: "PFCR = identity at init ($\gamma_l = 0$)".
- *Loss legend (bottom-right).* $\mathcal{L}_{\mathrm{seg}}$ attaches to the decoder outputs. $\mathcal{L}_{\mathrm{prior}}$ attaches to the prior head output. $\mathcal{L}_{\mathrm{rank}}$ attaches to the PC maps (dotted arrows out of each PFCR). $\mathcal{L}_{\mathrm{budget}}$ attaches to the route maps $R_l$ (dotted arrows out of each PFCR).
- *Color code.* Pretrained parts are blue, new parts orange or green, and loss connections dashed grey.

**Fig. 2: The PFCR module (full page width). It has two horizontal lanes that merge at the right.**

- *Input.* "$X_l$, $C_l\times D\times H\times W$" on the left.
- *Top lane (phase evidence).*
  1. $1\times1\times1$ conv $C_l\to1$ (init: channel mean) gives $p_l$, $1\times D\times H\times W$.
  2. An optional box "$\downarrow 2$ if $>2^{20}$ voxels" (dashed outline).
  3. A box "rFFT (fp32)".
  4. A kernel-bank box with three radial log-Gabor profiles $G_s(\rho)$, plotted over $\rho \in [0, 0.5]$ and annotated "learnable $\rho_s, \sigma_s$". Each profile is multiplied by $\{1,\ i u_z/\rho,\ i u_y/\rho,\ i u_x/\rho\}$, giving 12 kernels.
  5. A box "batched irFFT" giving the "$\mathbf{v}_s=(e_s,o_{s,z},o_{s,y},o_{s,x})$, $3\times4$ maps".
  6. Two parallel boxes: "$E=\lVert\sum_s\mathbf{v}_s\rVert$" and "$\Sigma_A=\sum_s\lVert\mathbf{v}_s\rVert$".
  7. A side box "noise threshold $T$": median$(A_1)/\sqrt{\ln 4}$ (stop-grad), $\times g$, $\times(\sqrt{\pi/2}+k\sqrt{(4-\pi)/2})$, with $k=\mathrm{softplus}(\theta^k)$.
  8. A box "PC $=[E-T]_+/(\Sigma_A+\epsilon\overline{\Sigma_A})$" giving "$\mathrm{PC}_l$, $1\times D\times H\times W$, $\in[0,1]$". Add "$\uparrow$" if the input was downsampled.
  9. A concat node with "$P_l$, $1\times D\times H\times W$".
  10. "Conv $3^3$, 2→8", then "ReLU", then "Conv $1^3$, 8→3 (zero-init)", then "softmax".
  11. The output is "$R_l$, $3\times D\times H\times W$". It then passes through "trilinear $\downarrow2$" to give "$\bar R_l$, $3\times\frac D2\times\frac H2\times\frac W2$".
- *Bottom lane (frequency content).*
  1. "Haar DWT" of $X_l$ gives 8 small cubes, each $C_l\times\frac D2\times\frac H2\times\frac W2$, in three color-coded groups. The low group, LLL, is green. The mid group, LLH/LHL/HLL, is yellow. The high group, LHH/HLH/HHL/HHH, is red.
  2. Three multiplier nodes connect the groups to the corresponding channels $\bar r^{\mathrm{low}}, \bar r^{\mathrm{mid}}, \bar r^{\mathrm{high}}$ of $\bar R_l$.
  3. "Haar IDWT" gives "$Y_l$, $C_l\times D\times H\times W$".
- *Merge.* A gain node "$\times\tanh(\gamma_l)$, $\gamma_l^{(0)}=0$" feeds a "$\oplus$" node that also takes a skip arrow from $X_l$. The output is "$\tilde X_l$, $C_l\times D\times H\times W$".
- *Side outputs.* Dotted arrows run from $\mathrm{PC}_l$ to "$\mathcal{L}_{\mathrm{rank}}$" and from $R_l$ to "$\mathcal{L}_{\mathrm{budget}}$".
- *Inset (bottom-left).* A 2D slice schematic of one lesion showing the boundary shell $B$ (2 voxels thick, across the contour), the eroded interior $I$ and the ring $O$ at 2–3 voxels outside. Label $B$ "positives" and $I\cup O$ "negatives".

## Open questions for the authors

**Code vs. V2 configuration.** This section describes the V2 behavior specified in the brief. Each item below is a place where the current code differs. The code needs to change, or this text does.

1. **Prior activation.** `phaser/unet.py` (`tumor_prior = torch.sigmoid(small_logits[:, 1:2])`) and `PhaseFrequencyCoherenceRouting.forward` (`prior.sigmoid()`) apply the sigmoid twice. V2 (Eq. 21) uses `softmax(small_logits, 1)[:, 1:2]` once.
2. **Prior supervision timing.** In `train_step`, the small-head loss runs only when `in_phase2` (epoch ≥ 60), weighted by `ramp_b`, a 50-epoch ramp starting at epoch 60. V2 starts it at epoch 0 with the 5-epoch ramp of Eq. (28).
3. **Budget-loss timing.** `phase_frequency_budget_loss` is weighted by `ramp_b = _aux_ramp(phase2_start=60)`, so it is effectively zero until epoch 60 and reaches full weight at epoch 110. V2 activates it from epoch 0 with the 5-epoch ramp.
4. **Ramp formula.** The PC-ranking ramp in the code is `min((e+1)/5, 1)`, which is non-zero at epoch 0. `_aux_ramp` is `(e - start)/warmup`, which is zero at its start epoch. I wrote Eq. (28) with `(e+1)/5` for all three auxiliary terms. Please confirm, and use one helper for all three.
5. **Segmentation loss.** `_build_loss` always wraps DiceCE in `CombinedDiceCEFocalTverskyLoss`, adding Focal Tversky with α = 0.3, β = 0.7, γ = 4/3 and weight `PHASER_FT_WEIGHT`, default 0.15. The module docstring wrongly says 0.3. V2 uses DiceCE only.
6. **Weight decay.** `configure_optimizers` applies `weight_decay = 3e-5` to every group, including `rho_raw`, `sigma_raw`, `noise_k` and `residual_gate`. V2 excludes them, which requires a separate no-decay group with the PhASER learning rate. Should the proxy weights and router convolutions also be excluded? I assumed they are not.
7. **Budget loss on tumor-free samples.** For samples without tumor, the code still adds the lesion term, `|0 − β_in|`, a constant 1/3 per sample with zero gradient. This inflates the logged loss but does not change training. Eq. (25) uses an indicator. Please align the code, or accept the discrepancy.
8. **Other switches.** The V2 launch must keep all other auxiliary losses and modules at their default "off" values. These are boundary, spectral, lesion-weight, gate, inside-liver and boundary-weighted losses; PCSF; LWFD; ACG; LA-ACG; T-RAF; and CFF (`PHASER_FUSION=none`). Please confirm.
9. **Checkpoint.** The trainer additionally saves `checkpoint_best_tumor_dice.pth` and nnU-Net's EMA-best checkpoint. V2 reports final-epoch weights, so evaluation scripts must load `checkpoint_final.pth`.

**Discrepancies between the brief and the code that I resolved in favor of the code (please confirm):**

10. **Router kernel.** The brief writes `Conv3(ReLU(Conv3(·)))`. The code's second layer is a 1×1×1 convolution (`conv_nd(dims, 8, 3, 1)`), so Eq. (13) uses 3×3×3 followed by 1×1×1. If "Conv3" meant "3 output channels", the two agree.
11. **Prior head structure.** The brief says "two 3×3×3 convs". The code has three 3×3×3 convolutions ($\phi_1$, $\phi_2$, $\psi_1$), then instance norm, ReLU and a 1×1×1 classifier (Eqs. 19–20).

**Open technical decisions ([[V2-pending]]):**

12. **Voxel cap.** With `pc_max_voxels = 2^20`, the PC of the shallowest skip is computed at $64^3$ and upsampled. The design review (I2) measured a substantial change of the PC map from this. Keep the cap, raise it, or remove it?
13. **Hard vs. soft threshold.** Eq. (11) uses a hard threshold $[E-T]_+$. A soft threshold is an alternative. The design review (I1) found $T/\Sigma_A$ between 1.17 and 1.98 on validation patches.
14. **Noise-scale index.** The noise scale always uses `A[:, 0]`, the scale *initialized* as finest. Nothing enforces $\rho_1 > \rho_2 > \rho_3$ after training. Should we sort the scales, or use `argmax(rho_s)`?
15. **Gain exponent.** Eq. (9) uses the noise-gain exponent 1, inherited from Kovesi's estimator. For an isotropic 3D band-pass filter, the white-noise amplitude grows roughly as $\rho_s^{3/2}$, so Eq. (9) is a heuristic in 3D. Should we keep it and call it a heuristic, or change the exponent?
16. **Physical spacing.** The frequency grid is in voxel units (`d = 1.0`), so the "isotropic" filters are anisotropic in millimetres (1.0 vs. 0.78 mm; design review I6). I described the grid in voxel units. Should we change the code, or state this as a limitation?
17. **Prior gradient.** $P$ is not detached before the router, so routing gradients also train the prior head. Is this intended? It slightly mixes the roles of $\mathcal{L}_{\mathrm{prior}}$ and the router.
18. **Softmax routing.** Softmax routing cannot switch off enhancement at a voxel (Eq. 18; design review C3). I state this as a property in Section 3.3. Should Section 5 discuss it as a limitation?

**Content to supply:**

19. The expansion of the acronym "PhASER".
20. The origin of the budget vectors $\boldsymbol{\beta}^{\mathrm{in}}, \boldsymbol{\beta}^{\mathrm{out}}$ and of the loss weights 0.2 / 0.05 / 0.1. Were they fixed a priori, or tuned on the 21 validation cases? The code comments cite a synthetic pilot for $\lambda_{\mathrm{rank}} = 0.2$; that pilot is not among the numbers we may cite.
21. The `batch_dice` setting from the nnU-Net plans, for $\mathcal{L}_{\mathrm{DC}}$ in Eq. (22).
22. The backbone parameter count and GMAC, so that the overhead ($141{,}641$ parameters; prior head ≈ 188.6 GMAC per $128^3$ patch) can be expressed as a percentage. The prior head's full-resolution 3×3×3 convolutions dominate the added compute, and reviewers may ask about this.
23. **Citations.** The closed-form Rayleigh threshold (Eqs. 8–10, median of the finest-scale amplitude) matches Kovesi's reference implementation more closely than the derivation in the 1999 paper. The paper estimates $E(E^2)$ from filter energies and uses the median of the squared amplitude. I cited only `kovesi1999phase`. Should we add a key for Kovesi's code or for a later Kovesi paper? No new keys were added.
24. **Gated residuals.** Zero-gated residual insertion is common practice. I described it without a citation. Add one if desired, for example from the VoCo or adapter literature; no listed key fits exactly.
