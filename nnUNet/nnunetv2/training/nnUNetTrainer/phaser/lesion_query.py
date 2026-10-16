"""
SALT: Spectral-Anchored Lesion Transformer (docs/SALT模型设计.md).

A lesion-centric mask-classification head on top of the (VoComni-pretrained) nnU-Net PlainConvUNet.
The voxel-wise semantic head is kept; in addition a set of lesion queries predicts one mask and one
objectness score per lesion instance, so that the lesion-versus-hard-negative decision (vessel
cross-sections, liver capsule and fissures, parenchymal heterogeneity) is taken at object level rather
than voxel by voxel. Three frequency-domain ingredients:

spectral_anchors  (SSA, §2.1)
    Scale-normalised Laplacian-of-Gaussian bank evaluated in the Fourier domain with physical frequencies
    (cycles / mm, so the anisotropic voxel spacing is handled exactly):
        R_s = F^-1[ F(v) * (2 pi s |f|)^2 exp(-2 pi^2 s^2 |f|^2) ] = -s^2 lap(G_s * v),
    s = r / sqrt(3) for a ball of radius r (the zero crossing of the 3D LoG lies at sqrt(3) s). Maxima over
    space and scale (Lindeberg's automatic scale selection) give lesion candidates with a centre and a
    radius. Two evidence maps: the semantic tumour probability (catches lesions the semantic head sees
    only faintly) and the foreground-restricted, standardised CT (|R|, both polarities; catches lesions the
    semantic head does not see, and supplies hard negatives). Each candidate becomes an anchored query.

LesionQueryDecoder  (§2.2)
    Mask2Former-style masked cross-attention over three decoder levels (coarse to fine, round robin),
    query self-attention, FFN; per query an objectness logit and a mask embedding whose dot product with
    full-resolution pixel embeddings gives the lesion mask. Anchored queries start from the memory
    feature at their centre and may always attend inside a ball of 2 r around it.

spectral_moment_descriptor  (SMD, §2.4)
    Second moments of the power spectrum of each query's soft mask,
        T = sum_f f f^T |M(f)|^2 / sum_f |M(f)|^2 = (1 / 4 pi^2) sum_x grad M grad M^T / sum_x M^2
    (Parseval), whose eigenvalue pattern separates blobs (isotropic spectrum: lesions) from tubes (energy
    on a plane: vessels) and sheets (energy on a line: capsule, fissures). Fed (detached) to the
    objectness head.

Training: lesion instances = 26-connected components of the tumour label (>= min_voxels); Hungarian
matching per decoder layer with the Mask2Former cost (2 class, 5 BCE, 5 Dice) on sampled points, with
the same number of points per lesion ("every lesion counts") plus uniform background points.
Inference: lesion map P(v) = max_q sigmoid(z_q) sigmoid(m_q(v)), fused with the semantic tumour
probability inside the predicted foreground (fuse_tumour).
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment

DEFAULT_RADII_MM = (2.5, 4.0, 6.0, 9.0, 13.5, 20.0)
_S26 = ndi.generate_binary_structure(3, 3)


# ---------------------------------------------------------------------------
# Spectral scale-space anchors
# ---------------------------------------------------------------------------

def log_bank_response(v: torch.Tensor, spacing, radii_mm=DEFAULT_RADII_MM) -> torch.Tensor:
    """v [B, D, H, W]. Scale-normalised LoG responses -s^2 lap(G_s * v), s = r / sqrt(3), computed in the
    Fourier domain (physical frequencies, zero padding of 3 s_max against wrap-around).
    Returns [B, K, D, H, W] float32; positive at bright blobs."""
    B, D, H, W = v.shape
    sig = [r / math.sqrt(3.0) for r in radii_mm]
    pad = [int(math.ceil(3 * max(sig) / float(s))) for s in spacing]
    vp = F.pad(v.float(), (pad[2], pad[2], pad[1], pad[1], pad[0], pad[0]))
    Dp, Hp, Wp = vp.shape[1:]
    fz = torch.fft.fftfreq(Dp, d=float(spacing[0]), device=v.device)
    fy = torch.fft.fftfreq(Hp, d=float(spacing[1]), device=v.device)
    fx = torch.fft.rfftfreq(Wp, d=float(spacing[2]), device=v.device)
    f2 = fz[:, None, None] ** 2 + fy[None, :, None] ** 2 + fx[None, None, :] ** 2
    V = torch.fft.rfftn(vp, dim=(1, 2, 3))
    out = []
    for s in sig:
        h = (2 * math.pi * s) ** 2 * f2 * torch.exp(-2 * math.pi ** 2 * s ** 2 * f2)
        r = torch.fft.irfftn(V * h, s=(Dp, Hp, Wp), dim=(1, 2, 3))
        out.append(r[:, pad[0]:pad[0] + D, pad[1]:pad[1] + H, pad[2]:pad[2] + W])
    return torch.stack(out, 1)


def scale_space_peaks(R: torch.Tensor, radii_mm, spacing, k: int, min_response: float = 0.0,
                      valid: torch.Tensor = None, nms_factor: float = 1.0, max_candidates: int = 256):
    """R [K, D, H, W] (one sample). Local maxima over 3x3x3 space and the adjacent scales with
    R > min_response (and inside `valid`), strongest first; greedy suppression of a weaker peak whose centre
    lies closer than nms_factor * max(r_i, r_j) mm. Returns at most k tuples (z, y, x, radius_mm, response)."""
    if k <= 0:
        return []
    neg = torch.full_like(R[:1], float("-inf"))
    is_max = (R >= F.max_pool3d(R[None], 3, 1, 1)[0]) & (R > min_response)
    is_max &= (R >= torch.cat([R[1:], neg], 0)) & (R >= torch.cat([neg, R[:-1]], 0))
    if valid is not None:
        is_max &= valid[None]
    idx = torch.nonzero(is_max)
    if idx.numel() == 0:
        return []
    vals = R[is_max]
    top = torch.topk(vals, min(int(vals.numel()), max_candidates)).indices
    idx, vals = idx[top].cpu().numpy(), vals[top].cpu().numpy()
    sp = np.asarray(spacing, dtype=np.float64)
    centres = idx[:, 1:] * sp
    radii = np.asarray(radii_mm, dtype=np.float64)[idx[:, 0]]
    keep = []
    for i in range(len(idx)):
        if keep:
            d = np.linalg.norm(centres[keep] - centres[i], axis=1)
            if (d < nms_factor * np.maximum(radii[keep], radii[i])).any():
                continue
        keep.append(i)
        if len(keep) == k:
            break
    return [(int(idx[i, 1]), int(idx[i, 2]), int(idx[i, 3]), float(radii[i]), float(vals[i])) for i in keep]


@torch.no_grad()
def spectral_anchors(sem_prob: torch.Tensor, image: torch.Tensor, fg: torch.Tensor, spacing,
                     n_sem: int, n_img: int, radii_mm=DEFAULT_RADII_MM, min_sem_response: float = 0.02):
    """sem_prob, image [B, D, H, W]; fg [B, D, H, W] bool (predicted liver + tumour).
    Returns dict of tensors: centre [B, Na, 3] (voxel z, y, x), radius [B, Na] mm, response [B, Na],
    valid [B, Na] bool, source [B, Na] (0 semantic, 1 image); Na = n_sem + n_img, unused slots invalid."""
    B = sem_prob.shape[0]
    na = n_sem + n_img
    dev = sem_prob.device
    centre = torch.zeros(B, na, 3, device=dev)
    radius = torch.ones(B, na, device=dev)
    resp = torch.zeros(B, na, device=dev)
    valid = torch.zeros(B, na, dtype=torch.bool, device=dev)
    source = torch.cat([torch.zeros(n_sem, dtype=torch.long), torch.ones(n_img, dtype=torch.long)]).to(dev)
    source = source[None].repeat(B, 1)
    r_sem = log_bank_response(sem_prob, spacing, radii_mm) if n_sem else None
    if n_img:
        x = image.float()
        body = x > x.flatten(1).amin(1)[:, None, None, None] + 1e-4
        m = fg.clone()
        for b in range(B):
            if int(m[b].sum()) < 1000:
                m[b] = body[b]
        mu = torch.stack([x[b][m[b]].mean() if m[b].any() else x[b].mean() for b in range(B)])
        sd = torch.stack([x[b][m[b]].std() if int(m[b].sum()) > 1 else x[b].std() for b in range(B)])
        v = (x - mu[:, None, None, None]) / (sd[:, None, None, None] + 1e-6) * m
        r_img = log_bank_response(v, spacing, radii_mm).abs()
    for b in range(B):
        found = []
        if n_sem:
            found += [(p, 0) for p in scale_space_peaks(r_sem[b], radii_mm, spacing, n_sem, min_sem_response)]
        if n_img:
            found += [(p, 1) for p in scale_space_peaks(r_img[b], radii_mm, spacing, n_img, 0.0, valid=m[b])]
        slot = {0: 0, 1: n_sem}
        for (z, y, x_, r, val), src in found:
            j = slot[src]
            centre[b, j] = torch.tensor([z, y, x_], dtype=torch.float32, device=dev)
            radius[b, j], resp[b, j], valid[b, j] = r, val, True
            slot[src] += 1
    return dict(centre=centre, radius=radius, response=resp, valid=valid, source=source)


# ---------------------------------------------------------------------------
# Spectral moment descriptor
# ---------------------------------------------------------------------------

@torch.no_grad()
def spectral_moment_descriptor(m: torch.Tensor, spacing) -> torch.Tensor:
    """m [B, N, d, h, w] soft masks on a grid with voxel spacing `spacing` (mm). Power-spectrum second
    moments via Parseval from central-difference gradients; eigenvalues l1 >= l2 >= l3 (cycles/mm)^2.
    Returns [B, N, 4]: log(l1 + l2 + l3), l2 / l1, l3 / l1, log(1 + volume_mm3)."""
    m = m.float()
    g = torch.gradient(m, spacing=tuple(float(s) for s in spacing), dim=(2, 3, 4))
    t = torch.stack([torch.stack([(g[a] * g[c]).sum((2, 3, 4)) for c in range(3)], -1) for a in range(3)], -2)
    energy = (m * m).sum((2, 3, 4))[..., None, None]
    t = t / (energy + 1e-6) / (4 * math.pi ** 2)
    t = t + 1e-8 * torch.eye(3, device=m.device)
    lam = torch.linalg.eigvalsh(t).flip(-1).clamp_min(0)          # descending
    l1 = lam[..., 0] + 1e-8
    vol = m.sum((2, 3, 4)) * float(np.prod(spacing))
    return torch.stack([torch.log(lam.sum(-1) + 1e-8), lam[..., 1] / l1, lam[..., 2] / l1, torch.log1p(vol)], -1)


# ---------------------------------------------------------------------------
# Lesion query decoder
# ---------------------------------------------------------------------------

def sine_pe_3d(coords01: torch.Tensor, dim: int, temperature: float = 10000.0) -> torch.Tensor:
    """coords01 [..., 3] in [0, 1] (z, y, x) -> [..., dim] sine/cosine features, dim // 6 frequencies per
    axis, zero padding up to dim."""
    nf = max(dim // 6, 1)
    freq = temperature ** (torch.arange(nf, device=coords01.device, dtype=torch.float32) / nf)
    x = coords01.float()[..., None] * (2 * math.pi) / freq
    pe = torch.cat([x.sin(), x.cos()], -1).flatten(-2)
    if pe.shape[-1] < dim:
        pe = F.pad(pe, (0, dim - pe.shape[-1]))
    return pe[..., :dim]


def _grid01(size, device):
    axes = [(torch.arange(n, device=device, dtype=torch.float32) + 0.5) / n for n in size]
    return torch.stack(torch.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)


class MLP(nn.Module):
    def __init__(self, d_in, d_hidden, d_out, n_layers):
        super().__init__()
        dims = [d_in] + [d_hidden] * (n_layers - 1) + [d_out]
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
        return x


class _Attention(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        assert d % heads == 0
        self.h, self.dh = heads, d // heads
        # No key bias: it adds the same q . b_k to every score of a query, which the softmax removes.
        self.q, self.v, self.o = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)
        self.k = nn.Linear(d, d, bias=False)

    def forward(self, q_in, k_in, v_in, allowed=None):
        B, N, _ = q_in.shape
        T = k_in.shape[1]
        q = self.q(q_in).view(B, N, self.h, self.dh).transpose(1, 2)
        k = self.k(k_in).view(B, T, self.h, self.dh).transpose(1, 2)
        v = self.v(v_in).view(B, T, self.h, self.dh).transpose(1, 2)
        mask = allowed[:, None] if allowed is not None else None   # bool, True = attend
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.o(out.transpose(1, 2).reshape(B, N, -1))


class _Layer(nn.Module):
    def __init__(self, d, heads, ffn):
        super().__init__()
        self.cross, self.selfa = _Attention(d, heads), _Attention(d, heads)
        self.ffn = nn.Sequential(nn.Linear(d, ffn), nn.ReLU(inplace=True), nn.Linear(ffn, d))
        self.n1, self.n2, self.n3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)

    def forward(self, q, pos, mem, mem_pe, allowed):
        q = self.n1(q + self.cross(q + pos, mem + mem_pe, mem, allowed))
        q = self.n2(q + self.selfa(q + pos, q + pos, q))
        return self.n3(q + self.ffn(q))


class LesionQueryDecoder(nn.Module):
    def __init__(self, level_channels, mask_channels: int, spacing, hidden: int = 128, heads: int = 4,
                 layers: int = 6, ffn: int = 512, n_sem: int = 16, n_img: int = 16, n_free: int = 8,
                 mask_dim: int = 32, radii_mm=DEFAULT_RADII_MM, shape_descriptor: bool = False):
        super().__init__()
        self.spacing = tuple(float(s) for s in spacing)
        self.radii_mm = tuple(radii_mm)
        self.n_sem, self.n_img, self.n_free = n_sem, n_img, n_free
        self.n_anchor = n_sem + n_img
        self.n_queries = self.n_anchor + n_free
        if self.n_queries == 0:
            raise ValueError("LesionQueryDecoder needs at least one query")
        self.hidden, self.n_layers, self.n_levels = hidden, layers, len(level_channels)
        self.shape_descriptor = shape_descriptor
        self.input_proj = nn.ModuleList(nn.Conv3d(c, hidden, 1) for c in level_channels)
        self.level_embed = nn.Embedding(self.n_levels, hidden)
        self.mask_proj = nn.Conv3d(mask_channels, mask_dim, 1)
        if n_free:
            self.free_feat = nn.Embedding(n_free, hidden)
            self.free_pos = nn.Embedding(n_free, hidden)
        if self.n_anchor:
            self.type_embed = nn.Embedding(2, hidden)
            self.anchor_content = MLP(7, hidden, hidden, 2)
            self.pos_mlp = MLP(hidden, hidden, hidden, 2)
        self.layers = nn.ModuleList(_Layer(hidden, heads, ffn) for _ in range(layers))
        self.norm = nn.LayerNorm(hidden)
        self.class_head = nn.Linear(hidden, 1)
        self.mask_head = MLP(hidden, hidden, mask_dim, 3)
        if shape_descriptor:
            self.shape_mlp = MLP(4, hidden, hidden, 2)
            nn.init.zeros_(self.shape_mlp.layers[-1].weight)
            nn.init.zeros_(self.shape_mlp.layers[-1].bias)

    # -- queries ---------------------------------------------------------------
    def _scale_features(self, radius, response):
        z = torch.log(radius / self.radii_mm[0]) / math.log(self.radii_mm[-1] / self.radii_mm[0])
        k = torch.arange(3, device=radius.device, dtype=torch.float32)
        ang = math.pi * z[..., None] * (2.0 ** k)
        return torch.cat([ang.sin(), ang.cos(), torch.log1p(response.clamp_min(0))[..., None]], -1)

    def _initial_queries(self, B, mem_fine_vol, full_size, anchors, device):
        feats, poss = [], []
        if self.n_anchor:
            c01 = (anchors["centre"] + 0.5) / torch.tensor(full_size, device=device, dtype=torch.float32)
            pos = self.pos_mlp(sine_pe_3d(c01, self.hidden))
            grid = (c01 * 2 - 1)[..., [2, 1, 0]].view(B, -1, 1, 1, 3)
            samp = F.grid_sample(mem_fine_vol, grid, mode="bilinear", align_corners=False)[:, :, :, 0, 0]
            feat = samp.transpose(1, 2) + self.type_embed(anchors["source"]) + \
                self.anchor_content(self._scale_features(anchors["radius"], anchors["response"]))
            # An unused anchor slot keeps only its source embedding and no position.
            v = anchors["valid"][..., None]
            feat = torch.where(v, feat, self.type_embed(anchors["source"]))
            pos = torch.where(v, pos, torch.zeros_like(pos))
            feats.append(feat)
            poss.append(pos)
        if self.n_free:
            feats.append(self.free_feat.weight[None].expand(B, -1, -1))
            poss.append(self.free_pos.weight[None].expand(B, -1, -1))
        return torch.cat(feats, 1), torch.cat(poss, 1)

    def _balls(self, anchors, size, full_size, B, device):
        """[B, N, T] bool: token centres within max(2 r, 1.5 level voxels) mm of a valid anchor."""
        N = self.n_queries
        if not self.n_anchor:
            return None
        scale = torch.tensor([f / s for f, s in zip(full_size, size)], device=device, dtype=torch.float32)
        sp = torch.tensor(self.spacing, device=device, dtype=torch.float32)
        tok = (_grid01(size, device) * torch.tensor(full_size, device=device, dtype=torch.float32) - 0.5) * sp
        cen = (anchors["centre"] * sp)                                     # [B, Na, 3] mm
        d2 = (tok ** 2).sum(-1)[None, None] - 2 * cen @ tok.T + (cen ** 2).sum(-1)[..., None]
        rad = torch.maximum(2 * anchors["radius"], 1.5 * (scale * sp).max())
        ball = (d2 <= rad[..., None] ** 2) & anchors["valid"][..., None]
        if self.n_free:
            ball = torch.cat([ball, torch.zeros(B, self.n_free, ball.shape[-1], dtype=torch.bool, device=device)], 1)
        return ball

    # -- heads -----------------------------------------------------------------
    def _heads(self, q, pooled_masks, fine_spacing):
        qn = self.norm(q)
        me = self.mask_head(qn)
        masks = [torch.einsum("bnc,bct->bnt", me.detach(), pm) for pm in pooled_masks]
        cls_in = qn
        if self.shape_descriptor:
            m = torch.sigmoid(masks[-1]).view(q.shape[0], q.shape[1], *self._fine_size)
            cls_in = qn + self.shape_mlp(spectral_moment_descriptor(m, fine_spacing))
        return self.class_head(cls_in)[..., 0], me, [torch.sigmoid(m) >= 0.5 for m in masks]

    def forward(self, levels, mask_feat, anchors=None):
        dev = mask_feat.device
        with torch.autocast(device_type=dev.type, enabled=False):
            mf = self.mask_proj(mask_feat.float())
            B = mf.shape[0]
            full_size = tuple(mf.shape[2:])
            mems, pes, sizes, pooled = [], [], [], []
            mem_fine_vol = None
            for l, f in enumerate(levels):
                m = self.input_proj[l](f.float())
                sizes.append(tuple(m.shape[2:]))
                if l == len(levels) - 1:
                    mem_fine_vol = m
                mems.append(m.flatten(2).transpose(1, 2) + self.level_embed.weight[l])
                pes.append(sine_pe_3d(_grid01(sizes[-1], dev), self.hidden)[None])
                pooled.append(F.adaptive_avg_pool3d(mf.detach(), sizes[-1]).flatten(2))
            self._fine_size = sizes[-1]
            fine_spacing = tuple(sp * f / s for sp, f, s in zip(self.spacing, full_size, sizes[-1]))
            if self.n_anchor and anchors is None:
                raise ValueError("anchored queries need anchors")
            q, pos = self._initial_queries(B, mem_fine_vol, full_size, anchors, dev)
            balls = [self._balls(anchors, s, full_size, B, dev) for s in sizes] if self.n_anchor else [None] * len(sizes)
            logits, embeds = [], []
            cls, me, allowed = self._heads(q, pooled, fine_spacing)
            logits.append(cls)
            embeds.append(me)
            for i, layer in enumerate(self.layers):
                l = i % self.n_levels
                a = allowed[l]
                if balls[l] is not None:
                    a = a | balls[l]
                a = a | ~a.any(-1, keepdim=True)          # a query with an empty mask attends everywhere
                q = layer(q, pos, mems[l], pes[l], a)
                cls, me, allowed = self._heads(q, pooled, fine_spacing)
                logits.append(cls)
                embeds.append(me)
        return dict(logits=logits, mask_embed=embeds, mask_features=mf, anchors=anchors)


@torch.no_grad()
def lesion_map(out: dict, chunk: int = 8) -> torch.Tensor:
    """P(v) = max_q sigmoid(z_q) sigmoid(m_q(v)) from the last layer; [B, D, H, W] float32."""
    mf, me = out["mask_features"].float(), out["mask_embed"][-1].float()
    s = torch.sigmoid(out["logits"][-1].float())
    p = torch.zeros((mf.shape[0],) + tuple(mf.shape[2:]), device=mf.device)
    for i in range(0, me.shape[1], chunk):
        m = torch.sigmoid(torch.einsum("bnc,bcdhw->bndhw", me[:, i:i + chunk], mf))
        p = torch.maximum(p, (m * s[:, i:i + chunk, None, None, None]).amax(1))
    return p


def fuse_tumour(sem_logits: torch.Tensor, lesion_prob: torch.Tensor, w: float,
                liver_label: int = 1, tumour_label: int = 2) -> torch.Tensor:
    """Semantic logits [B, C, ...] -> log-probabilities with the tumour probability replaced by
    w * p_t + (1 - w) * P * (p_l + p_t); liver gets the rest of the foreground, background unchanged."""
    p = torch.softmax(sem_logits.float(), 1)
    fg = p[:, liver_label] + p[:, tumour_label]
    pt = w * p[:, tumour_label] + (1 - w) * lesion_prob.float() * fg
    p = p.clone()
    p[:, tumour_label] = pt
    p[:, liver_label] = (fg - pt).clamp_min(0)
    return torch.log(p.clamp_min(1e-6)).to(sem_logits.dtype)


# ---------------------------------------------------------------------------
# Targets, matching, loss
# ---------------------------------------------------------------------------

def lesion_instances(target: torch.Tensor, min_voxels: int = 20, tumour_label: int = 2):
    """target [B, 1, D, H, W] -> (list of int32 instance maps [D, H, W], counts); 26-connected components of
    the tumour label with >= min_voxels voxels, numbered 1..K."""
    lab = target[:, 0].detach().cpu().numpy()
    maps, counts = [], []
    for b in range(lab.shape[0]):
        comp, n = ndi.label(lab[b] == tumour_label, structure=_S26)
        if n:
            sizes = np.bincount(comp.ravel(), minlength=n + 1)
            keep = np.flatnonzero(sizes >= min_voxels)
            keep = keep[keep > 0]
            remap = np.zeros(n + 1, np.int32)
            remap[keep] = np.arange(1, len(keep) + 1)
            comp, n = remap[comp], len(keep)
        maps.append(comp.astype(np.int32))
        counts.append(int(n))
    return maps, counts


def sample_points(inst: np.ndarray, n_per_instance: int = 512, n_uniform: int = 4096, dilate: int = 3,
                  rng: np.random.Generator = None):
    """Uniform points over the patch plus n_per_instance points inside every lesion dilated by `dilate`
    voxels (Chebyshev). Returns (flat indices int64, instance label of each point)."""
    rng = rng or np.random.default_rng()
    idx = [rng.integers(0, inst.size, n_uniform)]
    for k, sl in enumerate(ndi.find_objects(inst), start=1):
        if sl is None:
            continue
        box = tuple(slice(max(s.start - dilate, 0), min(s.stop + dilate, n)) for s, n in zip(sl, inst.shape))
        m = ndi.binary_dilation(inst[box] == k, structure=_S26, iterations=dilate)
        zz, yy, xx = np.nonzero(m)
        sel = rng.integers(0, len(zz), n_per_instance)
        idx.append(np.ravel_multi_index((zz[sel] + box[0].start, yy[sel] + box[1].start, xx[sel] + box[2].start),
                                        inst.shape))
    idx = np.concatenate(idx).astype(np.int64)
    return idx, inst.ravel()[idx].astype(np.int64)


def _bce_cost(x, g):
    pos = F.binary_cross_entropy_with_logits(x, torch.ones_like(x), reduction="none")
    neg = F.binary_cross_entropy_with_logits(x, torch.zeros_like(x), reduction="none")
    return (pos @ g.T + neg @ (1 - g).T) / x.shape[1]


def _dice_cost(x, g):
    s = x.sigmoid()
    return 1 - (2 * s @ g.T + 1) / (s.sum(-1)[:, None] + g.sum(-1)[None] + 1)


def lesion_query_loss(out: dict, inst_maps, counts, weights=(2.0, 5.0, 5.0), no_object_weight: float = 0.1,
                      n_per_instance: int = 512, n_uniform: int = 4096, rng: np.random.Generator = None,
                      stats: dict = None) -> torch.Tensor:
    """Mask2Former set loss on sampled points, averaged over the decoder layers (deep supervision)."""
    mf = out["mask_features"]
    dev = mf.device
    B = mf.shape[0]
    flat = mf.float().flatten(2)
    feats, gts = [], []
    for b in range(B):
        idx, lbl = sample_points(inst_maps[b], n_per_instance, n_uniform, rng=rng)
        feats.append(flat[b][:, torch.from_numpy(idx).to(dev)])
        k = torch.arange(1, counts[b] + 1, device=dev)[:, None]
        gts.append((torch.from_numpy(lbl).to(dev)[None] == k).float())
    num_masks = max(sum(counts), 1)
    total = 0.0
    n_matched = 0
    for logits, me in zip(out["logits"], out["mask_embed"]):
        logits = logits.float()
        cls_t = torch.zeros_like(logits)
        cls_w = torch.full_like(logits, no_object_weight)
        l_bce = l_dice = logits.new_zeros(())
        for b in range(B):
            if counts[b] == 0:
                continue
            pred = me[b].float() @ feats[b]                      # [N, P]
            g = gts[b]
            with torch.no_grad():
                cost = weights[0] * (-torch.sigmoid(logits[b]))[:, None] + weights[1] * _bce_cost(pred, g) \
                    + weights[2] * _dice_cost(pred, g)
            rows, cols = linear_sum_assignment(cost.cpu().numpy())
            r = torch.as_tensor(rows, device=dev, dtype=torch.long)
            c = torch.as_tensor(cols, device=dev, dtype=torch.long)
            cls_t[b, r] = 1.0
            cls_w[b, r] = 1.0
            pm, gm = pred[r], g[c]
            l_bce = l_bce + F.binary_cross_entropy_with_logits(pm, gm, reduction="none").mean(1).sum()
            pr = pm.sigmoid()
            l_dice = l_dice + (1 - (2 * (pr * gm).sum(1) + 1) / (pr.sum(1) + gm.sum(1) + 1)).sum()
            n_matched += len(rows)
        l_cls = (cls_w * F.binary_cross_entropy_with_logits(logits, cls_t, reduction="none")).sum() / cls_w.sum()
        total = total + weights[0] * l_cls + weights[1] * l_bce / num_masks + weights[2] * l_dice / num_masks
    if stats is not None:
        stats.update(n_instances=int(sum(counts)), n_matched_last=int(n_matched / max(len(out["logits"]), 1)))
    return total / len(out["logits"])
